import sys
import os
sys.path.append('..')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, Runnable
from src.utils import RandomNameGenerator, save_to_json, load_from_json, assign_ids, LLMChatModule, create_path_if_not_exists, get_siblings
from src.display import DisplayIter

from langchain_core.output_parsers import StrOutputParser, ListOutputParser
from tqdm.notebook import tqdm
from operator import itemgetter

from importlib import import_module
from datetime import datetime

class TableOfContents:
    def __init__(self, sections, contents_path):

        self.sections = sections
        self.contents_path = contents_path

        self.parent_sections_num = max(0, len(self.sections) - 2)
    
    def _get_children_for_elem(self, parent, children, parent_id='id', child_id='parent_id'):
        parent_id = parent[parent_id]
        return [x for x in children if x[child_id] == parent_id]
    
    def _sort_by_id(self, x):
        return sorted([{**elem, "num": self._get_level_id(elem['id'])} for elem in x], key=lambda x: int(x['num']))
    
    def _extract_children_files(self, parent, children):
        filtered_children = self._get_children_for_elem(parent, children.copy(), parent_id='id', child_id='parent_id')
        filtered_children = self._sort_by_id(filtered_children)
        return [f"{self.contents_path}/{elem['id']}.json" for elem in filtered_children]
    
    def _set_elem_nesting_level(self, elem):
        output = elem.copy()
        output['nesting_level'] = len(output['id'].split('.'))
        return output

    @staticmethod
    def _get_level_id(id):
        return int(id.split('.')[-1])
    
    def _assign_parent_to_elem(self, elem, parents):

        if self._get_level_id(elem['curr_id'])==1:
            elem['parents'] += [self._set_elem_nesting_level(x) for x in self._get_children_for_elem(elem, parents.copy(), parent_id='curr_parent_id', child_id='id')]
            last_parent = elem['parents'][-1].copy()
            elem['curr_id'] = elem['parents'][-1]['id']
            if 'parent_id' in last_parent.keys():
                elem['curr_parent_id'] = elem['parents'][-1]['parent_id']
        return elem
    
    def assign_contents(self):

        display_sections = self.sections[-2].copy()
        content_sections = self.sections[-1].copy()
        
        display_sections = [{"parents": [self._set_elem_nesting_level(elem)], 
                            "curr_id": elem['id'], 
                            "curr_parent_id": elem['parent_id'] if 'parent_id' in elem.keys() else None,
                            "files": self._extract_children_files(elem, content_sections)} 
                                 for elem in display_sections]

        for i in range(self.parent_sections_num):
            parents = self.sections[-2-i-1].copy()
            display_sections = [self._assign_parent_to_elem(elem, parents) for elem in display_sections]

        return display_sections
    
    @staticmethod
    def _get_keys(elem, keys):
        return ' '.join([elem[key] for key in keys])

    def get_flat_sections(self, nesting_level, keys):
        display_sections = self.sections[nesting_level].copy()

        display_sections = [{"parents": [self._set_elem_nesting_level(elem)], 
                                  "curr_id": elem['id'], 
                                  "curr_parent_id": elem['parent_id'] if 'parent_id' in elem.keys() else None} 
                                 for elem in display_sections]

        for parents in self.sections[:nesting_level][::-1]:
            display_sections = [self._assign_parent_to_elem(elem, parents) for elem in display_sections]

        display_sections = sum([[self._get_keys(x, keys) for x in elem['parents'][::-1]] for elem in display_sections], [])

        return display_sections
    

class GuideBuilder(LLMChatModule):
    def __init__(self, model_def, general_config, json_fix_config, concepts_config, parts_config, chapter_config, content_config, intro_config):
        super(GuideBuilder, self).__init__(model_def)

        self.json_fix_config = json_fix_config
        self.concepts_config = concepts_config
        self.general_config = general_config
        self.parts_config = parts_config
        self.chapter_config = chapter_config
        self.content_config = content_config
        self.intro_config = intro_config

        self.general_config['save_path'] = os.path.join('../locals', self.general_config['project_name'], 'contents')

        create_path_if_not_exists(self.general_config['save_path'])

        self._set_json_fix_chain(self.json_fix_config)

        self.name_gen = RandomNameGenerator()

        self.sections = None
        self.current_nesting_level = 0
        self.target_nesting_level = self.general_config["chapter_nesting_level"]

        self.outputs = []
        self.content_path = None
        self.output_path = None
        self.intro_path = None

    def _parse_list(self, list_str) -> Runnable:
        try:
            return ListOutputParser().invoke(list_str)
        except:
            return RunnableLambda(lambda x: [x])
        
    def _save_outputs(self, outputs, prefix):
        save_path = os.path.join(self.general_config["save_path"], prefix + self.name_gen() + '.json')
        save_to_json(outputs, save_path)
        return save_path
        
    def gen_concepts(self):
        print('Generating Concepts...')

        prompts_def = [self._load_prompt(path) for path in self.concepts_config["prompt_paths"]]
        concepts_num = self.concepts_config["concepts_num"]
        output_prefix = self.concepts_config["output_prefix"]

        chain_inputs = self._list2dicts(prompts_def, "raw_prompt")
        chain_inputs = self._expand_by_any_elem(chain_inputs, concepts_num, "concepts_num")
        chain_inputs = self._expand_by_any_elem(chain_inputs, self.general_config["main_subject"], "main_subject")
        chain_inputs = self._expand_by_any_elem(chain_inputs, self.concepts_config["instructs"], "instructs")
        

        chain_inputs *= self.concepts_config["inputs_mult"]

        concepts_chain = (
            RunnableLambda(lambda x: self._gen_dynamic_chat_template(x, system_key='system', user_key='template')) 
            | self.llm 
            | self._parse_json
            | RunnableLambda(lambda x: output_prefix + str(x))
            )
        
        outputs = concepts_chain.batch(chain_inputs)
        self._save_outputs(outputs, "concepts-")
        return outputs

    def gen_parts(self, concepts=None):
        print('Generating Parts...')
        self.current_nesting_level = 0
        
        prompts_def = [self._load_prompt(path) for path in self.parts_config["prompt_paths"]]
        instructs = self.parts_config["instructs"]
        add_concepts = self.parts_config["add_concepts"]
        section_args = self.parts_config["section_args"]

        chain_inputs = self._list2dicts(prompts_def, "raw_prompt")
        chain_inputs = self._expand_by_any_elem(chain_inputs, self.general_config["main_subject"], "main_subject")
        if add_concepts:
            if concepts is None:
                concepts = self.gen_concepts()
        else:
            concepts = ''
        chain_inputs = self._expand_by_any_elem(chain_inputs, concepts, "concepts")
        chain_inputs = self._expand_by_any_elem(chain_inputs, instructs, "instructs")

        _validate_format = self._get_format_validator(section_args)

        parts_chain = (
            RunnableLambda(lambda x: self._gen_dynamic_chat_template(x, system_key='system', user_key='template'))
            | self.llm
            | self._parse_json
            | _validate_format
            | RunnableLambda(lambda x: assign_ids(x, key='id', format=str, starting_num=1))
        )

        outputs = parts_chain.batch(chain_inputs)
        self._save_outputs(outputs, "parts-")

        self.display_collector = DisplayIter([outputs], self._save_display_outputs)

    def _save_display_outputs(self):
        sections = sum(self.display_collector.options.copy(), [])
        self.outputs.append(sections.copy())
        self.sections = sections
        self.gen_chapters()

    def load_sections(self, path, nesting_level=0):
        self.sections = load_from_json(path)
        self.current_nesting_level = nesting_level
        
        if self.current_nesting_level > 0:
            self.sections = sum(self.sections, [])

    def load_outputs(self, path):
        self.outputs = load_from_json(path)
        self.sections = self.outputs[-1].copy()
        self.current_nesting_level = len(self.outputs) - 1

    def _get_sections_string(self, sections):
        sections = [f"{elem['id']} {elem['title']} {elem['subtitle']}" for elem in sections.copy()]
        sep = self.general_config["sections_sep"]
        return sep.join(sections)
    
    def _get_toc_input(self, keys, max_nesting):
        toc_gen = TableOfContents(self.outputs, None)
        nesting_level = min(max_nesting, self.current_nesting_level-1)
        toc = toc_gen.get_flat_sections(nesting_level=nesting_level, keys=keys)
        toc = self.general_config["sections_sep"].join(toc)
        return toc
    
    def gen_chapters(self, sections=None):

        self.current_nesting_level += 1

        if sections is None:
            sections = self.sections.copy()

        if self.current_nesting_level <= self.target_nesting_level:

            print(f"Generating Chapters... Nesting Level: {self.current_nesting_level}")
            
            prompts_def = [self._load_prompt(path) for path in self.chapter_config["prompt_paths"]]
            instructs = self.chapter_config["instructs"]
            display_choices = self.current_nesting_level <= self.chapter_config["levels_for_choice"]
            section_args = self.chapter_config["section_args"]


            chain_inputs = self._list2dicts(prompts_def, "raw_prompt")
            chain_inputs = self._expand_by_any_elem(chain_inputs, self.general_config["main_subject"], "main_subject")
            chain_inputs = self._expand_by_any_elem(chain_inputs, instructs, "instructs")

            if not display_choices:
                chain_inputs = chain_inputs[:1]


            current_sections = self._get_sections_string(self.sections)
            chain_inputs = self._expand_by_any_elem(chain_inputs, current_sections, "current_sections")

            toc = self._get_toc_input(self.chapter_config["toc_keys"], self.chapter_config["toc_max_nesting"])
            chain_inputs = self._expand_by_any_elem(chain_inputs, [toc], "toc")
            
            _validate_format = self._get_format_validator(section_args)
            
            generation_chain = (
                RunnableLambda(lambda x: self._gen_dynamic_chat_template(x, system_key='system', user_key='template'))
                | self.llm
                | self._parse_json
                | _validate_format
                | RunnableLambda(lambda x: assign_ids(x, key='id', format=str, starting_num=1))
            )

            chapters_chain = (
                RunnableParallel(
                    sections = generation_chain,
                    parent_info = RunnableLambda(lambda x: [{"parent_id": x['id'], "parent_title": x['title']}])
                )
                | RunnableLambda(lambda x: self._apply_add_parent_id(self._concat_dicts(x['sections'], x['parent_info'])))
            )

            if display_choices:
                print('Sequential processing...')
                all_outputs = []
                for section in tqdm(sections):
                    curr_chain_inputs = chain_inputs.copy()
                    curr_chain_inputs = self._concat_dicts(curr_chain_inputs, [section])
                    curr_outputs = chapters_chain.batch(curr_chain_inputs)
                    all_outputs.append(curr_outputs)
                    
                self._save_outputs(all_outputs, f"lv{self.current_nesting_level}-chapters-")
                self.display_collector = DisplayIter(all_outputs.copy(), self._save_display_outputs)
            else:
                print("Single batch processing...")
                chain_inputs = self._concat_dicts(chain_inputs, self.sections)
                if self.current_nesting_level > 1:
                    chain_inputs = get_siblings(chain_inputs.copy(), sep=self.general_config["sections_sep"], key='current_sections')
                outputs = chapters_chain.batch(chain_inputs)
                self._save_outputs(outputs, f"lv{self.current_nesting_level}-chapters-")
                outputs = sum(outputs, [])
                self.sections = outputs.copy()
                self.outputs.append(outputs.copy())
                self.gen_chapters()
        else:
            self.output_path = self._save_outputs(self.outputs, f"outputs-")
            print("Chapter generation completed.")

    def _prepare_path(self, path):
        path = os.path.join(self.general_config["save_path"], path)
        try:
            os.makedirs(path)
        except:
            pass
        return path
    
    def _gen_chunks(self, lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def gen_contents(self):

        generation_name = self.name_gen()
        self.content_path = self._prepare_path(generation_name)

        batch_size = self.content_config["batch_size"]

        chapters = get_siblings(self.sections.copy(), sep='\n', key='current_sections')
        chapters = self._add_static_dict_elem(chapters, self.general_config["main_subject"], 'main_subject')
        chapters = self._add_static_dict_elem(chapters, self.content_config["instructs"], 'instructs')

        toc = self._get_toc_input(self.content_config["toc_keys"], self.content_config["toc_max_nesting"])
        chapters = self._expand_by_any_elem(chapters, [toc], "toc")


        prompt_def = self._load_prompt(self.content_config["prompt_path"])

        chat_prompt = ChatPromptTemplate([
                    {"role": "system", "content": prompt_def["system"]},
                    {"role": "user", "content": prompt_def["template"]}
                ])

        content_chain = chat_prompt | self.llm | StrOutputParser()

        full_chain = RunnableParallel(
            id = itemgetter('id'),
            content = content_chain
        )

        chunks = list(self._gen_chunks(chapters, batch_size))

        for chunk in tqdm(chunks):
            outputs = full_chain.batch(chunk)
            for output in outputs:
                save_to_json(output['content'], os.path.join(self.content_path, output['id'] + '.json'))
        

    def _apply_add_parent_id(self, x):
        return [self._add_parent_id(elem) for elem in x]
    
    def _add_parent_id(self, x):
        x["id"] = x["parent_id"] + '.' + x["id"]
        return x

    def _gen_dynamic_chat_template(self, chain_inputs, system_key=None, user_key=None):
        templates = chain_inputs['raw_prompt']
        system = templates[system_key] if system_key else None
        user = templates[user_key] if user_key else None
        
        chat_prompt = ChatPromptTemplate([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ])
        
        inputs = chain_inputs.copy()
        inputs.pop('raw_prompt')

        return chat_prompt.invoke(inputs)
    
    def gen_intro(self):

        print('Generating Introduction...')

        prompt_def = self._load_prompt(self.intro_config["prompt_path"])
        
        chat_prompt = ChatPromptTemplate([
                    {"role": "system", "content": prompt_def["system"]},
                    {"role": "user", "content": prompt_def["template"]}
                ])

        intro_chain = chat_prompt | self.llm | StrOutputParser()

        current_date = datetime.now().strftime("%B-%Y")
        model_name = self.model_def["model_args"]["model"]
        output = intro_chain.invoke({"main_subject": self.general_config["main_subject"], "date": current_date, "model_version": model_name})
        
        self.intro_path = os.path.join(self.general_config["save_path"], 'intro-' + self.name_gen() + '.json')
        save_to_json(output, self.intro_path)
        print("Introduction generation completed.")


    def _list2dicts(self, x, key):
        return [{key: elem} for elem in x]
    
    @staticmethod
    def _add_key_value(dict_input, key, value):
        dict_output = dict_input.copy()
        dict_output[key] = value
        return dict_output
    
    def _add_static_dict_elem(self, x, y, key):
        return [self._add_key_value(elem, key, y) for elem in x]
    
    def _add_list_dict_elem(self, x, y, key):
        return [self._add_key_value(x_elem, key, y_elem) for x_elem in x for y_elem in y]
    
    def _concat_dicts(self, x, y):
        return [{**x_elem, **y_elem} for x_elem in x for y_elem in y]
    
    def _expand_by_any_elem(self, x, y, key):
        if isinstance(y, list):
            return self._add_list_dict_elem(x, y, key)
        else:
            return self._add_static_dict_elem(x, y, key)