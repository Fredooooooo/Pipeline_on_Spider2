from llm.LLM import QWQ
import json
from tqdm import tqdm
from configs.Instruction import TABLE_AUG_INSTRUCTION, SQL_GENERATION_INSTRUCTION
import sys
import os
from utils.simplified_schema import simplified, explanation_collection
from configs.Instruction import KEY_WORD_AUG_INSTRUCTION, CONDITION_AUG_INSTRUCTION, SELF_CORRECTION_PROMPT
sys.path.append(".")
from configs.config import model_path, cuda_visible
from utils.util import execute_sql

os.environ["HF_DATASETS_CACHE"] = model_path
os.environ["HF_HOME"] = model_path
os.environ["HF_HUB_CACHE"] = model_path
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

step1_sl_input = step1_sql_input = step1_sl_output = step1_sql_output = step2_elements_input = step2_sql_input = step2_elements_output = step2_sql_output = step3_sql_input = step3_sql_output = step4_sql_input = step4_sql_output = 0

model = QWQ()
step1_sql_output = model.count_tokens('/fred/Pipeline_on_Spider2/spider2-snow/baselines/rslsql/src/sql_log/preliminary_sql.txt')
step2_sql_output = model.count_tokens('/fred/Pipeline_on_Spider2/spider2-snow/baselines/rslsql/src/sql_log/step_2_information_augmentation.txt')
step3_sql_output = model.count_tokens('/fred/Pipeline_on_Spider2/spider2-snow/baselines/rslsql/src/sql_log/step_3_binary.txt')
step4_sql_output = model.count_tokens('/fred/Pipeline_on_Spider2/spider2-snow/baselines/rslsql/src/sql_log/final_sql.txt')
step1_sl_output = model.count_tokens('/fred/Pipeline_on_Spider2/spider2-snow/baselines/rslsql/src/schema_linking/LLM.json')
step2_elements_output = model.count_tokens('/fred/Pipeline_on_Spider2/spider2-snow/baselines/rslsql/src/information/augmentation.json')

print(f"Total output tokens for step1_sql_output): {step1_sql_output}")
print(f"Total output tokens for step2_sql_output): {step2_sql_output}")
print(f"Total output tokens for step3_sql_output): {step3_sql_output}")
print(f"Total output tokens for step4_sql_output): {step4_sql_output}")
print(f"Total output tokens for step1_sl_output): {step1_sl_output}")
print(f"Total output tokens for step2_elements_output): {step2_elements_output}")


def table_info_construct(ppl):
    (question, simple_ddl, ddl_data,
     foreign_key, evidence, example) = (ppl['question'].strip(), ppl['simplified_ddl'].strip(),
                                        ppl['ddl_data'].strip(), ppl['foreign_key'].strip(),
                                        ppl['evidence'].strip(), ppl['example'])

    table_info = ('### Sqlite SQL tables, with their properties:\n' + simple_ddl +
                  '\n### Here are some data information about database references.\n' + ddl_data +
                  '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key)
    return table_info


def table_column_selection(table_info, ppl):
    evidence = ppl['evidence'].strip()
    question = ppl['question'].strip()
    json_prompt = '''Respond only with valid json. Do not write an introduction or summary.
    The format is {"tables": ["table1", "table2", ...],"columns":["table1.`column1`","table2.`column2`",...]}
    '''
    prompt_table = table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question
    prompt = prompt_table + '\n\n' + json_prompt
    return model.count_tokens(prompt)


def preliminary_sql(table_info, ppl):
    example = ppl['example']
    evidence = ppl['evidence'].strip()
    question = ppl['question'].strip()
    json_prompt = '''Respond only with valid json. Do not write an introduction or summary.
    The format is {"sql": "SQL statement that meets the user's question requirements"}
    '''
    table_info = example.strip() + "\n\n### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question
    prompt = table_info + '\n\n' + json_prompt
    return model.count_tokens(prompt)


def step1(ppl_file, step1_sl_input, step1_sql_input, step1_sl_output, x=0):
    # 1.加载prompt信息 从0开始
    with open(ppl_file, 'r') as f:
        ppls = json.load(f)

    # for i in tqdm(range(x, len(ppls))):
    # Try first 128 questions for now
    for i in tqdm(range(x, 128)):
        information = {}
        ppl = ppls[i]

        # table_info
        table_info = table_info_construct(ppl)

        #  table_column
        step1_sl_input += table_column_selection(table_info, ppl)

        # preliminary_sql
        step1_sql_input += preliminary_sql(table_info, ppl)
    step1_sql_input += step1_sl_output
    print(f"Total prompt tokens for step1_sl_input): {step1_sl_input}")
    print(f"Total prompt tokens for step1_sql_input): {step1_sql_input}")

step1("src/information/ppl_dev.json", step1_sl_input, step1_sql_input, step1_sl_output)


def table_info_construct2(simple_ddl, ddl_data, foreign_key, explanation):
    table_info = ('### Sqlite SQL tables, with their properties:\n' + simple_ddl +
                  '\n### Here are some data information about database references.\n' + ddl_data +
                  '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key +
                  '\n### The meaning of every column:\n#\n' + explanation.strip() +
                  '\n#\n')

    return table_info


def table_augmentation(table_info, ppl):
    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    table_qwq_res_prompt = table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question
    json_prompt = '''Respond only with valid json. Do not write an introduction or summary.
    The format is {"tables": ["table1", "table2", ...],"columns":["table1.`column1`","table2.`column2`",...]}
    '''
    prompt = table_qwq_res_prompt + '\n\n' + json_prompt
    return model.count_tokens(prompt)


def key_word_augmentation(table_info, ppl):
    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    word_qwq_res_prompt = table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question
    json_prompt = '''Respond **only** in valid JSON format. Do not include any text outside of JSON. The JSON should look like this:
    {"sql_keywords": ["keyword1", "keyword2", ...]}
    '''
    prompt = word_qwq_res_prompt + '\n\n' + json_prompt
    return model.count_tokens(prompt)


def condition_augmentation(ppl):
    question = ppl['question'].strip()
    json_prompt = '''Respond **only** in valid JSON format. Do not include any text outside of JSON. The JSON should look like this:
    {"conditions": ["condition1", "condition2", ...]}
    '''
    prompt = question + '\n\n' + json_prompt
    return model.count_tokens(prompt)


def sql_generation(ppl, table_info):
    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    example = ppl['example']

    table_info = example.strip() + '\n\n' + "### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question
    json_prompt = '''Respond **only** in valid JSON format. Do not include any text outside of JSON. The JSON should look like this:
    {"sql": "SQL statement that meets the user's question requirements"}
    '''
    prompt = table_info + '\n\n' + json_prompt
    return model.count_tokens(prompt)


def step2(ppl_file, step2_elements_input, step2_sql_input, step2_elements_output, x=0):
    # 1.加载prompt信息 从0开始
    with open(ppl_file, 'r') as f:
        ppls = json.load(f)

    # for i in tqdm(range(x, len(ppls))):
    # Try first 128 questions for now
    for i in tqdm(range(x, 128)):
        information = {}
        ppl = ppls[i]

        # 简化ddl
        simple_ddl, ddl_data, foreign_key = simplified(ppl)

        # 列描述
        explanation = explanation_collection(ppl)

        # table_info
        table_info = table_info_construct2(simple_ddl, ddl_data, foreign_key, explanation)

        # table_aug
        step2_elements_input += table_augmentation(table_info, ppl)

        # word_aug
        step2_elements_input += key_word_augmentation(table_info, ppl)

        # condition_aug
        step2_elements_input += condition_augmentation(ppl)

        # sql_generation
        step2_sql_input += sql_generation(ppl, table_info)
    step2_sql_input += step2_elements_output
    print(f"Total prompt tokens for step2_elements_input): {step2_elements_input}")
    print(f"Total prompt tokens for step2_sql_input): {step2_sql_input}")

step2("src/information/ppl_dev.json", step2_elements_input, step2_sql_input, step2_elements_output)


def prompt_construct(simple_ddl, ddl_data, foreign_key, explanation, ppl, sql1, sql2):
    db = ppl['db']
    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    example = ppl['example']

    table_info = '### Sqlite SQL tables, with their properties:\n'
    table_info += simple_ddl + '\n' + '### Here are some data information about database references.\n' + ddl_data + '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key + '\n### The meaning of every column:\n#\n' + explanation.strip() + "\n#\n"

    table_info = example.strip() + '\n\n' + "### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question

    r1, c1, re1 = execute_sql(sql1, db)
    r2, c2, re2 = execute_sql(sql2, db)

    candidate_sql = f"### sql1: {sql1} \n### result1: {re1} \n### sql2: {sql2} \n### result2: {re2}"

    return table_info, candidate_sql


def step3(ppl_file, sql_file1, sql_file2, step3_sql_input, x=0):
    # 1.加载prompt信息 从0开始
    with open(ppl_file, 'r') as f:
        ppls = json.load(f)

    with open(sql_file1, 'r') as f:
        sqls1s = f.readlines()

    with open(sql_file2, 'r') as f:
        sqls2s = f.readlines()

    # for i in tqdm(range(x, len(ppls))):
    # Try first 128 questions for now
    for i in tqdm(range(x, 128)):
        ppl = ppls[i]
        sql1 = sqls1s[i].strip()
        sql2 = sqls2s[i].strip()

        # 简化ddl
        simple_ddl, ddl_data, foreign_key = simplified(ppl)

        # 列描述
        explanation = explanation_collection(ppl)

        table_info, candidate_sql = prompt_construct(simple_ddl, ddl_data, foreign_key, explanation, ppl, sql1, sql2)

        step3_sql_input += model.count_tokens(table_info + candidate_sql)
    print(f"Total prompt tokens for step3_sql_input): {step3_sql_input}")

step3("src/information/ppl_dev.json", "src/sql_log/preliminary_sql.txt", "src/sql_log/step_2_information_augmentation.txt", step3_sql_input)


def table_info_construct4(ppl, simple_ddl, ddl_data, foreign_key, explanation):
    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    example = ppl['example']

    table_info = '### Sqlite SQL tables, with their properties:\n'
    table_info += simple_ddl + '\n' + '### Here are some data information about database references.\n' + ddl_data + '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key + '\n### The meaning of every column:\n#\n' + explanation.strip() + "\n#\n"

    table_info += f'\n### sql_keywords: {ppl["sql_keywords"]}'
    table_info += f'\n### conditions: {ppl["conditions"]}'

    table_info = example.strip() + '\n\n' + "### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n' + '### Hint: ' + evidence + "\n### Question: " + question + '\n\n' + 'The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.'

    return table_info


def step4(ppl_file, sql_file, step4_sql_input, x=0):
    # 1.加载prompt信息 从0开始
    with open(ppl_file, 'r') as f:
        ppls = json.load(f)

    with open(sql_file, 'r') as f:
        pre_sqls = f.readlines()

    sys_message = SELF_CORRECTION_PROMPT

    # for i in tqdm(range(x, len(ppls))):
    # Try first 128 questions for now
    for i in tqdm(range(x, 128)):
        message = []
        message.append({'role': 'system', 'content': sys_message})
        ppl = ppls[i]
        db = ppl['db']

        # 简化ddl
        simple_ddl, ddl_data, foreign_key = simplified(ppl)

        # 列描述
        explanation = explanation_collection(ppl)

        table_info = table_info_construct4(ppl, simple_ddl, ddl_data, foreign_key, explanation)

        pre_sql = pre_sqls[i].strip()

        num = 0
        while num < 5:

            row_count, column_count, result = execute_sql(pre_sql, db)

            if num > 0:
                table_info = "### Buggy SQL: " + pre_sql.strip() + "\n" + f"### The result of the buggy SQL is [{result.strip()}]. Please fix the SQL to get the correct result."
            if row_count == 0 and column_count == 0:
                json_prompt = '''Respond only with valid json. Do not write an introduction or summary.
                The format is {"sql": "SQL statement that meets the user question requirements"}
                '''
                prompt = table_info + '\n\n' + json_prompt
                message.append({'role': 'user', 'content': prompt})
                step4_sql_input += model.count_tokens(prompt)
                num += 1
            else:
                break
    print(f"Total prompt tokens for step4_sql_input: {step4_sql_input}")

step4("src/information/ppl_dev.json", "src/sql_log/final_sql.txt", step4_sql_input)