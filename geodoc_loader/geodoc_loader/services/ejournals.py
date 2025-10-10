import time
import numpy as np
from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sys
from datetime import datetime, timedelta

# GCP
import firebase_admin
from firebase_admin import firestore

from google.cloud import storage, bigquery
from google.auth import default

#only for downloader
import requests
import os
import fitz
import io
from PIL import Image

from geodoc_config import load_config_by_path
from geodoc_loader.download.gcp import upload_dicts_to_bigquery_table


class RandomPause:
    def __init__(self, pause_mean, pause_std, pause_min):
        self.pause_mean = pause_mean
        self.pause_std = pause_std
        self.pause_min = pause_min
    
    def __call__(self,):
            '''
                Proceed random pause from normal distribution with given parameters
            ''' 
            pause_time = max(self.pause_min, np.random.normal(loc=self.pause_mean, scale=self.pause_std, size=None))
            time.sleep(pause_time)

class EJournalCrawler:
    def __init__(self, 
                 province_id, 
                 pause_mean, 
                 pause_std, 
                 pause_min, 
                 scrolling_limit, 
                 loading_timeout,
                 ignore_first_page_image, 
                 image_dpi, 
                 browser_size_x, 
                 browser_size_y, 
                 browser_timeout,
                 dataset_id,
                 document_bucket_name,
                 images_bucket_name,
                 firestore_collection_name,
                 status_table,
                 logs_table,
                 errors_table,
                 ejournals_urls_path,
                 ejournals_urls_filename
                 ):
        '''
            province_id: there will be web address downloaded from firestore for one of province official 
                        journal like 'https://edzienniki.duw.pl'
            
            pause_mean, pause_std: parameters for normal distridution to generate random pause (in seconds)
            pause_min: minimal value of pause time (in seconds)
            
            scrolling_limit: [positive integer] limit of scrolls down to get to bottom of page 
            (if too low  crawler might not get to the bottom)
            
            loading_timeout: seconds to wait until page is loaded
            image_dpi: dpi for PDFs page screenshot
            
            ignore_first_page_image: do not include first page when extracting images from documents
        '''

        # assign gcp locations
        self.dataset_id = dataset_id
        self.document_bucket_name = document_bucket_name
        self.images_bucket_name = images_bucket_name
        self.firestore_collection_name = firestore_collection_name
        self.status_table = status_table
        self.logs_table = logs_table
        self.errors_table = errors_table
        
        # Set up connection to Firestore
        self.app = firebase_admin.initialize_app()
        self.db = firestore.client()

        # Get web path for given province
        self.province_id = province_id
        self.web_path = load_config_by_path(ejournals_urls_path, ejournals_urls_filename)[province_id]['web_path']

        # Set up connection to BigQuery
        _, self.project_id = default()
        self.client = bigquery.Client()
        self.status_table_id = f'{self.project_id}.{dataset_id}.{status_table}'
        
        # limits and pauses
        self.scrolling_limit = scrolling_limit
        self.loading_timeout = loading_timeout
        
        self.random_pause = RandomPause(pause_mean, pause_std, pause_min)
        
        # Launches Selenium driver on Firefox browser and open selected webpage
        self.display = Display(visible=0, size=(browser_size_x, browser_size_y))
        self.display.start()
        self.driver = webdriver.Firefox()
        self.driver.set_window_size(browser_size_x, browser_size_y)
        self.driver.set_page_load_timeout(browser_timeout)

        self.driver.get(self.web_path + '/actbymonths')
        
        # Available months and years options
        self.random_pause()
        self.months_list = self.get_dropdown_items('month')
        self.years_list = self.get_dropdown_items('year')
        
        # Arguments for row parser
        self.info_class_names = {
                    'type': ("span", "type ng-binding"),
                    'signature': ("span", "nr ng-binding"), 
                    'publisher': ("span", "publisher ng-binding ng-scope"),
                    'text_date': ("span", "day ng-binding"),
                    'publish_date': ("td", "acts__publish-date ng-binding"),
                    'act_position': ("td", "acts__position ng-binding"),
                    'act_subject': ("a", "subject ng-binding")
                   }
        
        self.ignore_first = ignore_first_page_image
        
        self.months_names = ['Styczeń','Luty','Marzec','Kwiecień','Maj','Czerwiec','Lipiec','Sierpień','Wrzesień',
              'Październik','Listopad','Grudzień']
        
        # get buckets for saving files to Cloud Storage
        self.storage_client = storage.Client(project=self.project_id)
        self.pdf_bucket = self.storage_client.bucket(self.project_id + '-' + 'pdfs')
        self.img_bucket = self.storage_client.bucket(self.project_id + '-' + 'page_imgs')
        
        self.dpi = image_dpi
    
    
    def get_dropdown_items(self, dropdown_id):
        '''
            dropdown_id: id of dropdown list from which we want to get items from
            
            Returns list of strings with available options
        '''
        select_box = Select(self.driver.find_element(By.ID, dropdown_id))

        return [element.get_attribute("text") for element in select_box.options]
    
    
    def scroll_to_bottom(self,):
        '''
            Scroll down until bottom of the page or get to defined iteration limit
        '''
        self.random_pause()
        curr_height = self.driver.execute_script("return document.body.scrollHeight;")
        for i in range(self.scrolling_limit):
            last_height = curr_height
            self.driver.execute_script("window.scrollTo(0, {curr_height});".format(curr_height=curr_height))
            
            self.random_pause()
            
            curr_height = self.driver.execute_script("return document.body.scrollHeight;")
            if last_height==curr_height:
                break
        
        return last_height==curr_height
        
    def select_dropdown_item(self, dropdown_id, item_text):
        '''
            Select option for given dropdown list
        '''
        self.random_pause()
        select = Select(self.driver.find_element(By.ID,dropdown_id))
        select.select_by_visible_text(item_text)
        
    def use_page_filter(self, filter_text):
        '''
            Uses text filter on page
            
            ID of textbox is given dynamically so we have to get it from attribute of another element
        '''
        # Extract ID of text input
        searchbox_pos = self.driver.find_elements(By.CLASS_NAME,"sr-only")[7]
        searchbox_id = searchbox_pos.get_attribute("for")
        # Get text box element
        searchbox = self.driver.find_element(By.ID,searchbox_id)
        # Put letters one by one
        searchbox.clear()
        for letter in filter_text:
            # type letter
            searchbox.send_keys(letter)
            # random break after each letter
            time.sleep(np.random.uniform(0.1,0.8))
        self.random_pause()

    def update_record(self, cr):
        update_query = f"""
            UPDATE {self.status_table_id}
            SET 
                completed = {cr['completed']},
                last_call = '{cr['last_call']}',
                acts_founded = {cr['acts_founded']},
                acts_downloaded = {cr['acts_downloaded']}
            WHERE
                id = '{cr['id']}'
        """

        query_job = self.client.query(update_query)
        result = query_job.result()  # Wait for the query to complete

        # Check if any rows were affected
        if query_job.num_dml_affected_rows > 0:
            print(f"Successfully updated {query_job.num_dml_affected_rows} record(s) in BigQuery.")
            return True
        else:
            print("No records were updated in BigQuery.")
            return False
    
    @staticmethod
    def is_month_finished(query_year, query_month):
        current_date = datetime.now()
        diff = (current_date.year-int(query_year)) * 12 + (current_date.month-int(query_month))
        if diff>0:
            return True
        else:
            return False
                
    def parse_page(self, year, month, text_filter=None):
        '''
            Set specified year and month from dropdown list and scroll to bottom of page
            
            year, month: integers specyfing acts publishing dates (months numbers from 1 to 12)
            
            text_filter (Optional): string to filter results on page
            
            Returns status
        '''
        print('\n' + '-'*50)
        print(f'Parsing page for province {self.province_id} [y-m] {year}-{month}')
        start_time = time.time()
        year_name = str(year)
        month_name = self.months_names[month-1]
        self.acts_downloaded = 0

        queue_id = '_'.join([self.province_id, str(year), str(month)])

        errors = []
        parsed_rows = []
        
        if year_name not in self.years_list:
            return None
        elif month_name not in self.months_list:
            return None
        else:

            # Get downloaded documents for queue_id
            try:
                fs_doc_resp = self.db.collection('acts').where('queue_id', '==', queue_id).get()
                existing_docs = {doc.id: doc.to_dict() for doc in fs_doc_resp}
                self.acts_downloaded = len(existing_docs)
            except Exception as e:
                err_msg = f'Error getting existing docs from Firestore: {str(e)}'
                print(err_msg)
                errors.append(err_msg)
                return None, errors

            # Wait until page is loaded
            try:
                WebDriverWait(self.driver, self.loading_timeout) \
                    .until(EC.presence_of_element_located((By.ID, 'month')))
                '''for i in range(self.loading_repeats):
                    time.sleep(1)
                    if self.driver.find_element(By.ID, 'month')!=None:
                        break'''
                        
                # Select options for given parameters
                self.select_dropdown_item('month',month_name)
                self.select_dropdown_item('year', year_name)
                
                # Filter results
                if text_filter:
                    self.use_page_filter(text_filter)
                
                scroll_result = self.scroll_to_bottom()
            except Exception as e:
                err_msg = f'Error loading page for {self.province_id} [y-m] {year}-{month}: {str(e)}'
                print(err_msg)
                errors.append(err_msg)
                return None, errors
            
            for _ in range(2):
                self.random_pause()
            
            # Find all Acts on page
            try:
                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                body = soup.find('tbody')
                rows = body.find_all("tr", class_="pointer ng-scope")
                acts_founded = len(rows)
            except Exception as e:
                err_msg = f'Error parsing page for {self.province_id} [y-m] {year}-{month}: {str(e)}'
                print(err_msg)
                errors.append(err_msg)
                return None, errors
            
            # Iterate over rows and extract data
            for row_id, row in enumerate(rows):
                parsed_row, error = self.parse_row(row, year, existing_docs=existing_docs, queue_id=queue_id, row_id=row_id)
                if error:
                    errors.append(error)
                    continue
                parsed_rows.append(parsed_row)

            print('Acts founded: %d' % (acts_founded) | ' Acts downloaded: %d' % (self.acts_downloaded))

            ## Update status table in BigQuery
            # Check if query month is finished
            completed = self.is_month_finished(year, month)

            if acts_founded>self.acts_downloaded:
                completed = False

            update_info = {'completed': completed,
                           'last_call': str(datetime.now()),
                           'acts_founded': acts_founded,
                           'acts_downloaded': self.acts_downloaded,
                           'id': queue_id}
            
            update_result = self.update_record(update_info)

            if not update_result:
                err_msg = f'Error updating status table in BigQuery for {self.province_id} [y-m] {year}-{month}'
                print(err_msg)
                errors.append(err_msg)
            else:
                print(f'Successfully updated status table in BigQuery for {self.province_id} [y-m] {year}-{month}')

            ###

            ## Log info to BigQuery table

            end_time = time.time()
            print(f'Page parsing time: {end_time-start_time} seconds')

            logs_info = {
                'id': queue_id,
                'call_data': str(datetime.now()),
                'docs_processed': len(rows),
                'docs_founded': acts_founded,
                'docs_downloaded': self.acts_downloaded,
                'errors_num': len(errors),
                'processing_time': end_time-start_time
            }

            logs_upload = upload_dicts_to_bigquery_table(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                table_name=self.logs_table,
                data=[logs_info]
            )

            if not logs_upload:
                err_msg = f'Error uploading logs to BigQuery table for {self.province_id} [y-m] {year}-{month}'
                print(err_msg)
                errors.append(err_msg)
            else:
                print(f'Successfully uploaded logs to BigQuery table for {self.province_id} [y-m] {year}-{month}')

            ###

            ## Log errors to BigQuery table
            if len(errors)>0:
                errors = [{"id": queue_id, "error_message": err, "timestamp": str(datetime.now())} for err in errors]
                errors_upload = upload_dicts_to_bigquery_table(
                    project_id=self.project_id,
                    dataset_id=self.dataset_id,
                    table_name=self.errors_table,
                    data=errors
                )
                if not errors_upload:
                    err_msg = f'Error uploading errors to BigQuery table for {self.province_id} [y-m] {year}-{month}'
                    print(err_msg)
                else:
                    print(f'Successfully uploaded {len(errors)} errors to BigQuery table for {self.province_id} [y-m] {year}-{month}')
            else:
                print(f'No errors to upload to BigQuery table for {self.province_id} [y-m] {year}-{month}')
            ###
            print('-'*50 + '\n')

            return parsed_rows
    
    def get_pages_with_images(self, pdf_bytes):
        '''
            Takes 'screenshots' of pages assumed to contain map images

            Return list of bytes images
        '''
        starting_page = int(self.ignore_first)

        # Read PDF from bytes
        pdf_file = fitz.open(stream=pdf_bytes, filetype='pdf')

        map_pages = []
        pages_nums = []
        # Iterate over pages by page number (optional: ingore first page)
        for page_index in range(starting_page, len(pdf_file)):
            page = pdf_file[page_index]
            # Return list of images metadata
            page_imgs_info = page.get_image_info()
            # If list is empty then there is no image
            if len(page_imgs_info)>0:
                # take page screenshot and add to the list
                page_img = page.get_pixmap(dpi=self.dpi).tobytes()
                map_pages.append(page_img)
                pages_nums.append(page_index+1)

        return map_pages, pages_nums
        
    def parse_row(self, row_object, year, existing_docs=None, queue_id=None, row_id=None):
        '''
            Extract important data from given row of ejournal page
            Also downloads attached PDF file and store it as bytes

            row_object: bs4.element.Tag type containing single act data
            info_class_names: python dict where each key contains tuple of arguments for find function

            Returns python dict with extracted texts and full document
        '''
        
        #### EXTRACT INFO FROM PAGE FOR SINGLE ACT ####
        
        parsed_info = {}
        err_msg = None
        # Iterate over keys from info_class_names dict
        try:
            for key in self.info_class_names.keys():
                elem, class_name = self.info_class_names[key]
                # extract and trim string for key argmunets
                found_elem = row_object.find(elem, class_=class_name)
                if found_elem!=None:
                    found_elem = found_elem.get_text().replace(u'\xa0', u' ').strip()
                parsed_info[key] = found_elem
        except Exception as e:
            err_msg = f'Error parsing row {row_id} info class names for {self.province_id} [y] {year}: {str(e)}'
            print(err_msg)
            return None, err_msg

        # Additional data not included in class_info
        try:
            found_elem = row_object.find("td", class_="acts__date") \
                                    .find("span", class_="ng-binding")
            if found_elem!=None:
                found_elem = found_elem.get_text().strip()
            parsed_info['act_date'] = found_elem
            parsed_info['act_link'] = urljoin(self.web_path,
                                            row_object.find("a", class_="btn btn-link btn-sm ng-scope").attrs['href'])
            parsed_info['year'] = year
            parsed_info['last_update'] = datetime.now()

            doc_name = '_'.join([self.province_id, str(year), parsed_info['act_position']])
        except Exception as e:
            err_msg = f'Error parsing row {row_id} additional data for {self.province_id} [y] {year}: {str(e)}'
            print(err_msg)
            return None, err_msg
        
        ####

        #### CHECK IF IT HAS BEEN ALREADY DOWNLOADED ####

        fs_doc_resp = existing_docs.get(doc_name, None)

        # downloaded flag indicates whether PDF is to be downloaded again (if false)
        if fs_doc_resp:
            downloaded = fs_doc_resp['downloaded']
        else:
            downloaded = False
        
        # if PDF has been already downloaded ignore further processing
        if not downloaded:

            ####
            
            #### DOWNLOAD ACTs ATTACHED PDF FILE ####
            
            url = parsed_info['act_link']
            try:
                # get content from link
                pdf_bytes = requests.get(url).content
                downloaded = True
                self.acts_downloaded += 1
            except Exception as e:
                err_msg = f'Error downloading PDF for row {row_id} for {self.province_id} [y] {year}: {str(e)}'
                print(err_msg)
                downloaded = False
            
            # Add info about download status
            parsed_info['downloaded'] = downloaded
            
            if downloaded:
                # upload PDF to Cloud Storage
                parsed_info['pdf_name'] = doc_name
                
                try:
                    blob = self.pdf_bucket.blob(doc_name)
                    blob.upload_from_string(data=pdf_bytes, content_type='application/pdf')
                except Exception as e:
                    err_msg = f'Error uploading PDF to Cloud Storage for row {row_id} for {self.province_id} [y] {year}: {str(e)}'
                    print(err_msg)
                    downloaded = False
                    parsed_info['downloaded'] = downloaded
                    parsed_info['pdf_name'] = None

            parsed_info['queue_id'] = queue_id
            
            ####
            
            #### UPLOAD DOC INFO TO FIRESTORE
            try:
                self.db.collection('acts').document(doc_name).set(parsed_info)
            except Exception as e:
                err_msg = f'Error uploading doc info to Firestore for row {row_id} for {self.province_id} [y] {year}: {str(e)}'
                print(err_msg)
                return None, err_msg
                
            ####
        return parsed_info
    
    def quit_driver(self,):
        '''
            Shutdown Selenium driver and closes all browser windows
        '''
        self.driver.quit()
        self.display.stop()