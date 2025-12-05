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
    """
        Class to generate random pauses based on normal distribution
    """
    def __init__(self, pause_mean, pause_std, pause_min):
        self.pause_mean = pause_mean
        self.pause_std = pause_std
        self.pause_min = pause_min
    
    def __call__(self,):
        """
        Proceed random pause from normal distribution with given parameters
        """
        pause_time = max(self.pause_min, np.random.normal(loc=self.pause_mean, scale=self.pause_std, size=None))
        time.sleep(pause_time)

class EJournalCrawler:
    """
    Class to crawl ejournals website for given province and download acts documents
    """
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
                 documents_bucket_name,
                 images_bucket_name,
                 firestore_collection_name,
                 status_table,
                 logs_table,
                 errors_table,
                 ejournals_urls_path,
                 ejournals_urls_filename
                 ):
        """
        Initialize EJournalCrawler instance.

        Args:
            province_id (str): Identifier for the province to crawl.
            pause_mean (float): Mean pause time in seconds.
            pause_std (float): Standard deviation of pause time in seconds.
            pause_min (float): Minimum pause time in seconds.
            scrolling_limit (int): Maximum number of scrolls to reach the bottom of the page.
            loading_timeout (int): Timeout in seconds for page loading.
            ignore_first_page_image (bool): Whether to ignore the first page when extracting images.
            image_dpi (int): DPI for PDF page screenshots.
            browser_size_x (int): Width of the browser window.
            browser_size_y (int): Height of the browser window.
            browser_timeout (int): Timeout in seconds for browser operations.
            dataset_id (str): GCP BigQuery dataset ID.
            documents_bucket_name (str): GCP Cloud Storage bucket name for documents.
            images_bucket_name (str): GCP Cloud Storage bucket name for images.
            firestore_collection_name (str): Firestore collection name for storing document metadata.
            status_table (str): BigQuery table name for status tracking.
            logs_table (str): BigQuery table name for logging.
            errors_table (str): BigQuery table name for error logging.
            ejournals_urls_path (str): Path to the configuration file with ejournal URLs.
            ejournals_urls_filename (str): Filename of the configuration file with ejournal URLs.
        """

        # assign gcp locations
        self.dataset_id = dataset_id
        self.documents_bucket_name = documents_bucket_name
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
        self.pdf_bucket = self.storage_client.bucket(self.project_id + '-' + documents_bucket_name)
        self.img_bucket = self.storage_client.bucket(self.project_id + '-' + images_bucket_name)
        
        self.dpi = image_dpi
    
    
    def get_dropdown_items(self, dropdown_id):
        """
        Gets the list of items from a dropdown menu by its ID.

        Args:
            dropdown_id (str): The ID of the dropdown element.
        Returns:
            list: List of item texts in the dropdown.
        """
        select_box = Select(self.driver.find_element(By.ID, dropdown_id))

        return [element.get_attribute("text") for element in select_box.options]
    
    
    def scroll_to_bottom(self,):
        """
        Scroll down until bottom of the page or get to defined iteration limit

        Args:
            None
        Returns:
            bool: True if reached bottom of the page, False if limit reached.
        """
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
        """
        Selects an item from a dropdown menu by its visible text.

        Args:
            dropdown_id (str): The ID of the dropdown element.
            item_text (str): The visible text of the item to select.
        Returns:
            None
        """
        self.random_pause()
        select = Select(self.driver.find_element(By.ID,dropdown_id))
        select.select_by_visible_text(item_text)
        
    def use_page_filter(self, filter_text):
        """
        Uses the search box on the page to filter results by given text.

        Args:
            filter_text (str): Text to filter the results.
        Returns:
            None
        """

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
        """
        Update a record in the status table in BigQuery.

        Args:
            cr (dict): Dictionary containing the record information to update.
        Returns:
            bool: True if the update was successful, False otherwise.
        """
        update_query = f"""
            UPDATE {self.status_table_id}
            SET 
                completed = {cr['completed']},
                last_call = '{cr['last_call']}',
                acts_found = {cr['acts_found']},
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
        """
        Check if the queried month is finished compared to the current date.

        Args:
            query_year (str): Year of the queried month.
            query_month (str): Month of the queried month.
        Returns:
            bool: True if the queried month is finished, False otherwise.
        """
        current_date = datetime.now()
        diff = (current_date.year-int(query_year)) * 12 + (current_date.month-int(query_month))
        if diff>0:
            return True
        else:
            return False
                
    def parse_page(self, year, month, text_filter=None):
        """
        Parse the ejournal page for a given year and month, download acts documents, and update status in BigQuery.
        Parsing process includes:
            1. Check existing downloaded documents in Firestore.
            2. Load and filter the page for the given year and month.
            3. Find all acts on the page.
            4. Iterate over acts, extract data, download PDFs, and upload to Cloud Storage and Firestore.
            5. Update status table in BigQuery.

        Args:
            year (int): Year to parse.
            month (int): Month to parse.
            text_filter (str, optional): Text filter to apply on the page. Defaults to None.
        Returns:
            list: List of parsed rows with act information.
        """
        print('\n' + '-'*50)
        print(f'Parsing page for province {self.province_id} [y-m] {year}-{month}')
        start_time = time.time()
        year_name = str(year)
        month_name = self.months_names[month-1]
        self.acts_downloaded = 0

        queue_id = '_'.join([self.province_id, str(year), str(month)])

        errors = []
        parsed_rows = []
        
        if (year_name not in self.years_list) or (month_name not in self.months_list):
            acts_founded = 0
            rows = []
        else:

            # Get downloaded documents for queue_id
            try:
                fs_doc_resp = self.db.collection(self.firestore_collection_name).where('queue_id', '==', queue_id).get()
                if len(fs_doc_resp)>0:
                    print(f'Found {len(fs_doc_resp)} existing documents in Firestore for {self.province_id} [y-m] {year}-{month}')
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
                print(f'Acts found on page: {acts_founded}')
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

            print(f'Acts founded: {acts_founded} | Acts downloaded: {self.acts_downloaded}')

        ## Update status table in BigQuery
        # Check if query month is finished
        completed = self.is_month_finished(year, month)

        if acts_founded>self.acts_downloaded:
            completed = False

        update_info = {'completed': completed,
                        'last_call': str(datetime.now()),
                        'acts_found': acts_founded,
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
            'call_date': str(datetime.now()),
            'docs_processed': len(rows),
            'docs_found': acts_founded,
            'docs_downloaded': self.acts_downloaded,
            'errors_num': len(errors),
            'processing_time': round(end_time-start_time,1)
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
        """
        Extracts images from PDF bytes, returning images for each page.
        Args:
            pdf_bytes (bytes): The PDF file content in bytes.
        Returns:
            list: List of images in bytes for each page containing images.
        """
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
        """
        Parse a single row (act) from the ejournal page, download the attached PDF document,
        upload it to Cloud Storage, and store metadata in Firestore.

        Args:
            row_object (bs4.element.Tag): BeautifulSoup Tag object representing the row.
            year (int): Year of the act.
            existing_docs (dict, optional): Dictionary of existing documents in Firestore to avoid re-downloading. Defaults to None.
            queue_id (str, optional): Queue identifier for the current parsing session. Defaults to None.
            row_id (int, optional): Row index for logging purposes. Defaults to None.
        Returns:
            tuple: (parsed_info (dict), err_msg (str or None))
        """
        
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
                self.db.collection(self.firestore_collection_name).document(doc_name).set(parsed_info)
            except Exception as e:
                err_msg = f'Error uploading doc info to Firestore for row {row_id} for {self.province_id} [y] {year}: {str(e)}'
                print(err_msg)
                return None, err_msg
                
            ####
        return parsed_info, err_msg
    
    def quit_driver(self,):
        """
        Quit the Selenium WebDriver and stop the virtual display.
        """
        self.driver.quit()
        self.display.stop()

from geodoc_config import get_service_config
from geodoc_loader.download.queue import get_queue_items_query
from geodoc_loader.download.bigquery import get_query_result
from geodoc_loader.download.process import filter_queue_by_worker

def run_ejournals_download():
    """
    Run the ejournals downloader service.
    This function retrieves queue items from BigQuery, initializes the EJournalCrawler for each province,
    and processes the pages for the specified year and month, downloading acts documents as needed.

    Function uses environment variables for configuration:
        PAUSE_MEAN (float): Mean pause time in seconds.
        PAUSE_STD (float): Standard deviation of pause time in seconds.
        PAUSE_MIN (float): Minimum pause time in seconds.
        SCROLLING_LIMIT (int): Maximum number of scrolls to reach the bottom of the page.
        LOADING_TIMEOUT (int): Timeout in seconds for page loading.
        IGNORE_FIRST_PAGE_IMAGE (bool): Whether to ignore the first page when extracting images.
        IMAGE_DPI (int): DPI for PDF page screenshots.
        QUEUE_TIMEOUT (int): Timeout in seconds for queue operations.
        BROWSER_SIZE_X (int): Width of the browser window.
        BROWSER_SIZE_Y (int): Height of the browser window.
        BROWSER_TIMEOUT (int): Timeout in seconds for browser operations.
        QUEUE_LIMIT (int): Maximum number of queue items to process.
        FILTER (str): Additional filter to apply to the queue items.

    Returns:
        None
    """
    pause_mean = float(os.getenv("PAUSE_MEAN", 1.5))
    pause_std = float(os.getenv("PAUSE_STD", 0.7))
    pause_min = float(os.getenv("PAUSE_MIN", 0.5))
    scrolling_limit = int(os.getenv("SCROLLING_LIMIT", 20))
    loading_timeout = int(os.getenv("LOADING_TIMEOUT", 60))
    ignore_first_page_image = os.getenv("IGNORE_FIRST_PAGE_IMAGE", "True").lower() in ("true", "1", "yes")
    image_dpi = int(os.getenv("IMAGE_DPI", 120))
    queue_timeout = int(os.getenv("QUEUE_TIMEOUT", 60))
    text_filter = os.getenv("TEXT_FILTER", "zagosp")
    browser_size_x = int(os.getenv("BROWSER_SIZE_X", 1920))
    browser_size_y = int(os.getenv("BROWSER_SIZE_Y", 1080))
    browser_timeout = int(os.getenv("BROWSER_TIMEOUT", 60))

    config = get_service_config(service_name="ejournals-downloader", key="worker")
    queue_limit = int(os.environ.get("QUEUE_LIMIT", 1))
    filter = os.environ.get("FILTER", None)

    full_filter = "completed = false" + (f" AND {filter}" if filter else "")

    bq_client = bigquery.Client()
    project_id = bq_client.project

    queue_query = get_queue_items_query(
        project_id=project_id,
        dataset_id=config['dataset_id'],
        table_id=config['status_table'],
        limit=queue_limit,
        desc_priority=False,
        filter=full_filter,
        order_by='province_id'
    )

    queue_items = get_query_result(client=bq_client, query=queue_query)
    queue_items = filter_queue_by_worker(queue_items)
    queue_items_num = len(queue_items)

    PROVINCE_ID = ''
    print(f'Used TEXT_FILTER: {text_filter}')

    for i, item in enumerate(queue_items):
        if item['province_id']!=PROVINCE_ID:
            # if it is not first input given during worker session shutdown firebase and selenium drivers
            if PROVINCE_ID!='':
                firebase_admin.delete_app(crw.app)
                crw.quit_driver()
            PROVINCE_ID = item['province_id']
            print('PROVINCE_ID: %s' % (PROVINCE_ID))
            crw = EJournalCrawler(province_id=PROVINCE_ID, 
                            pause_mean=pause_mean, 
                            pause_std=pause_std, 
                            pause_min=pause_min, 
                            scrolling_limit=scrolling_limit,
                            loading_timeout=loading_timeout,
                            ignore_first_page_image=ignore_first_page_image,
                            image_dpi=image_dpi,
                            browser_size_x=browser_size_x, 
                            browser_size_y=browser_size_y, 
                            browser_timeout=browser_timeout,
                            dataset_id=config['dataset_id'],
                            documents_bucket_name=config['documents_bucket_name'],
                            images_bucket_name=config['images_bucket_name'],
                            firestore_collection_name=config['firestore_collection_name'],
                            status_table=config['status_table'],
                            logs_table=config['logs_table'],
                            errors_table=config['errors_table'],
                            ejournals_urls_path=config['ejournals_urls_path'],
                            ejournals_urls_filename=config['ejournals_urls_filename']
                            )
        parsed_rows = crw.parse_page(
            year=item['year'],
            month=item['month'],
            text_filter=text_filter
        )

        print(f'Processed {i+1} out of {queue_items_num} queue items')