import sys
import os
sys.path.append('..')

import markdown2
import pdfkit
import fitz

from src.utils import save_to_json, load_from_json, create_path_if_not_exists
from src.builder import TableOfContents
from src.cover import VoronoiCover
from tqdm.notebook import tqdm

def get_file_list(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except Exception as e:
        return []
    
def load_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    return html_content

def concatenate_pdfs(pdf_paths, output_path):
    # Create a new PDF document
    output_pdf = fitz.open()

    # Iterate through the list of PDF paths
    for pdf_path in pdf_paths:
        # Open the current PDF
        current_pdf = fitz.open(pdf_path)
        
        # Append each page of the current PDF to the output PDF
        for page_num in range(len(current_pdf)):
            output_pdf.insert_pdf(current_pdf, from_page=page_num, to_page=page_num)
    
    # Save the concatenated PDF to the specified output path
    output_pdf.save(output_path)
    print(f"Concatenated PDF saved to: {output_path}")

class Markdown2PDF:
    def __init__(self, project_name, contents_path, chapters_file, header_path, footer_path, main_part_template, sub_part_template, formatting, toc_path, intro_file, title, cover_args, logo_path, footer_logo_path, section_limit=None):

        self.project_name = project_name
        self.contents_path = os.path.join('../locals', self.project_name, 'contents', contents_path)
        self.chapters_path = os.path.join('../locals', self.project_name, 'contents', chapters_file)
        self.save_path = os.path.join('../locals', self.project_name, 'chapters')
        self.header_path = header_path
        self.footer_path = footer_path
        self.main_part_template = main_part_template
        self.sub_part_template = sub_part_template
        self.formatting = formatting
        self.toc_path = toc_path
        self.intro_path = os.path.join('../locals', self.project_name, 'contents', intro_file)
        self.title = title
        self.cover_args = cover_args
        self.logo_path = logo_path
        self.footer_logo_path = footer_logo_path

        self.section_limit = section_limit

        self.head_page_num = 0

        self.header = load_html_file(self.header_path).format(**self.formatting)
        self.footer = load_html_file(self.footer_path)
        self.toc_template = load_html_file(self.toc_path)

        create_path_if_not_exists(self.save_path)

        self.sections = load_from_json(self.chapters_path)
        self.contents_files = get_file_list(self.contents_path)

        self.toc = TableOfContents(self.sections, self.contents_path)
        self.display_sections = None

        self.cover_image_generator = VoronoiCover(**cover_args)
    
    def _assign_contents(self):

        self.display_sections = self.toc.assign_contents()

        if self.section_limit:
            self.display_sections = self.display_sections[:self.section_limit]

    def _write_chapters(self):
        for section in tqdm(self.display_sections):
            self._write_single_chapter(section)

        save_to_json(self.display_sections, os.path.join(self.save_path, 'display_sections.json'))

    def _write_single_chapter(self, section):
        
        file_name = section['curr_id']

        html_content = ""
        # add section titles
        for parent in section['parents'][::-1]:
            template = self.main_part_template if parent['nesting_level'] == 1 else self.sub_part_template
            html_content += markdown2.markdown(template.format(**parent))
        # add subchapters contents
        for file_path in section['files']:
            content = load_from_json(file_path)
            html_content += markdown2.markdown(content, extras=['tables', 'fenced-code-blocks'])

        # Combine header, TOC, content, and footer
        full_html =  self.header + html_content + self.footer

        html_path = os.path.join(self.save_path, f"{file_name}.html")
        pdf_path = os.path.join(self.save_path, f"{file_name}.pdf")
        # Save the HTML content to a file
        self._save_html(html_path, full_html)

        self._save_pdf(html_path, pdf_path)

        # Delete the HTML file after converting to PDF
        os.remove(html_path)

        numpages = self._get_pdf_pages_num(pdf_path)

        section['num_pages'] = numpages

    def _save_pdf(self, html_path, pdf_path):
        # Convert HTML to PDF with headers and footers
        pdfkit.from_file(html_path, pdf_path, options={
            'enable-local-file-access': None,
            'disable-external-links': None,
            'enable-forms': None,
            'disable-smart-shrinking': None,
            'zoom': 0.75,
            'load-error-handling': 'ignore',
            'no-images': None,
            'encoding': 'UTF-8',
            'page-size': 'A4',
            'margin-top': '2cm',
            'margin-bottom': '2cm',
            'margin-left': '2cm',
            'margin-right': '2cm',
            'dpi': 150
        })

    def _save_html(self, html_path, html_content):
        with open(html_path, "w") as html_file:
            html_file.write(html_content)

    def _generate_toc_entry_html(self, id, title, page_number, level):
        if level==1:
            line = f"""<div class="title_div"><span class="title" style='margin-left: {level*50}px'>{title}</span></div>"""
        else:
            line = f"""<div class="wrap">
            <span class="left" style='margin-left: {(level-1)*50}px'>{id} {title}</span>
            <span class="right", style='margin-right: 150px'>{page_number}</span>
            </div>"""
        return line
    
    def _save_toc(self):
        toc_elems = self.display_sections.copy()
        curr_page_num = self.head_page_num
        toc_content_html = []

        for section in toc_elems:
            for toc_elem in section['parents'][::-1]:
                toc_content_html.append(self._generate_toc_entry_html(title=toc_elem['title'], page_number=curr_page_num+2, level=toc_elem['nesting_level'], id=toc_elem['id']))
            curr_page_num += section['num_pages']

        toc_content_html = self.toc_template.format(toc_entries='\n'.join(toc_content_html), **self.formatting)

        html_path = os.path.join(self.save_path, "toc.html")
        pdf_path = os.path.join(self.save_path, "toc.pdf")

        self._save_html(html_path, toc_content_html)
        self._save_pdf(html_path, pdf_path)
        os.remove(html_path)

        return pdf_path
    
    @staticmethod
    def _get_pdf_pages_num(path):
        doc = fitz.open(path)
        return doc.page_count

    def _generate_toc(self):
        print('Generating Table of Contents...')
        pdf_path = self._save_toc()

        numpages = self._get_pdf_pages_num(pdf_path)

        self.head_page_num += numpages

        if numpages > 1:
            pdf_path = self._save_toc()

    def _generate_intro(self):

        print('Generating Introduction...')
        intro_content = load_from_json(self.intro_path)
        intro_content = markdown2.markdown(intro_content, extras=['tables', 'fenced-code-blocks'])

        html_content = self.header + intro_content + self.footer

        html_path = os.path.join(self.save_path, "intro.html")
        pdf_path = os.path.join(self.save_path, "intro.pdf")

        self._save_html(html_path, html_content)
        self._save_pdf(html_path, pdf_path)

        os.remove(html_path)

        numpages = self._get_pdf_pages_num(pdf_path)

        self.head_page_num += numpages

    def _generate_cover(self):
        print('Generating Cover...')
        cover_img_path = os.path.join(self.save_path, "cover.png")
        self.cover_image_generator(cover_img_path)

        # Create a new PDF document
        pdf_document = fitz.open()

        # Add a blank page
        page = pdf_document.new_page(width=595.28, height=841.89)

        # Insert the background image
        page.insert_image(fitz.Rect(0, 0, page.rect.width, page.rect.height), filename=cover_img_path, overlay=False)

        # Define the title text and its properties
        title_text = '\n'.join(self.title.split(' '))
        title_font_size = 64
        title_color = (0.9, 0.9, 0.9)  # White color in RGB

        # Define the position for the title text
        title_position = fitz.Point(50, 250)  # Adjust the position as needed

        # Add the title text to the page
        page.insert_text(title_position, title_text, fontsize=title_font_size, color=title_color, fontname="ubuntubo")

        img_size = 100
        padding = 15

        prect = page.rect
        center = prect.width/2
        y1 = prect.height - padding  # bottom of footer rect
        y0 = y1 - img_size  # top of footer rect
        header_rect = fitz.Rect(center-img_size/2, y0, center+img_size/2, y1)  # height 20 points
        page.insert_image(header_rect, filename=self.logo_path, overlay=True)

        pdf_path = os.path.join(self.save_path, "cover.pdf")
        # Save the PDF document
        pdf_document.save(pdf_path, garbage=4, deflate=True, deflate_images=True, deflate_fonts=True)
        pdf_document.close()

    def _concat_pages(self):
        print('Concatenating pages...')
        cover_path = os.path.join(self.save_path, "cover.pdf")
        intro_path = os.path.join(self.save_path, "intro.pdf")
        toc_path = os.path.join(self.save_path, "toc.pdf")

        sections = self.display_sections.copy()

        doc = fitz.open()
        for head_path in [cover_path, intro_path, toc_path]:
            doc.insert_pdf(fitz.open(head_path), from_page=0, to_page=-1)

        curr_page_num = head_pages = doc.page_count

        img_size = 50
        padding = 5

        for section in tqdm(sections):
            section_path = os.path.join(self.save_path, f"{section['curr_id']}.pdf")
            section_doc = fitz.open(section_path)

            for page in section_doc:
                prect = page.rect
                center = prect.width/2
                header_rect = fitz.Rect(center-img_size/2, padding, center+img_size/2, padding+img_size)  # height 20 points
                page.insert_image(header_rect, filename=self.footer_logo_path, overlay=True)

                ftext = str(curr_page_num)
                y1 = prect.height - 20  # bottom of footer rect
                y0 = y1 - 20  # top of footer rect
                footer_rect = fitz.Rect(0, y0, prect.width, y1)  # rect has full page width
                page.insert_textbox(footer_rect, ftext, align=fitz.TEXT_ALIGN_CENTER, fontsize=8)

                curr_page_num += 1

            doc.insert_pdf(section_doc, from_page=0, to_page=-1)


        toc_list = []
        curr_page_num = head_pages + 1

        toc_list.append((1, 'Important Information', 2))
        toc_list.append((1, 'Table of Contents', 3))

        for section in self.display_sections:
            toc_list.extend([self._get_toc_info(x, curr_page_num) for x in section['parents'][::-1]])
            curr_page_num += section['num_pages']

        doc.set_toc(toc_list)

        doc.save(os.path.join(self.save_path, f"{self.title}.pdf"), garbage=4, deflate=True, deflate_images=True, deflate_fonts=True)

    @staticmethod
    def _get_toc_info(elem, page):
            title = elem['title'] if elem['nesting_level'] == 1 else f"{elem['id']} {elem['title']}"
            return (elem['nesting_level'], title, page)