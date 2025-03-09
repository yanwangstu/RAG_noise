import json
import time
import argparse
from unitProcess import questionAdjust
from unitProcess import documentGeneration
from unitProcess import answerAdjust
from unitProcess import spaceDelete


def data_process(filePathRead, filePathWrite):
    generated_item = 0
    count = -1
    # 后期需要修改为异步调用 以提高running speed
    with (open(filePathRead, 'r') as fileRead, open(filePathWrite, 'w') as fileWrite):
        fileWrite.write('[\n')
        for line in fileRead:
            count += 1
            try:
                json_line = json.loads(line)
                document_tokens = json_line['document_tokens']
                # long_answer_candidates_range is 2d
                long_answer_candidates_range = [[item["start_token"], item["end_token"]]
                                                for item in json_line['long_answer_candidates']]
                # long_answer_range is 1d
                long_answer_range = [json_line["annotations"][0]["long_answer"]["start_token"],
                                     json_line["annotations"][0]["long_answer"]["end_token"]]
                # short_answer_range is 2d
                short_answers_range = [[item["start_token"], item["end_token"]]
                                       for item in json_line["annotations"][0]["short_answers"]]
                if short_answers_range != []:
                    # generate the question
                    origin_question = json_line["question_text"]
                    question = questionAdjust.question_adjust(origin_question)

                    # generate the answer 1d list
                    answers = [answerAdjust.answer_adjust(document_tokens, answer_range)
                               for answer_range in short_answers_range]

                    # generate the golden documents
                    golden_document = documentGeneration.document_generation(document_tokens, long_answer_range)
                    golden_documents = documentGeneration.golden_documents_generation(question, golden_document, answers)
                    golden_documents.append(golden_document)

                    # generate the distracting_documents
                    distracting_document = documentGeneration.distracting_documents_generation(question, golden_document, answers)

                    # generate counterfactual and inconsequential documents
                    transfer_documents, counterfactual_documents, inconsequential_documents = documentGeneration.counterfactual_inconsequential_documents_generation(
                        question, document_tokens, long_answer_candidates_range, long_answer_range)

                    # generate other documents
                    # other_document = [documentGeneration.document_generation(document_tokens, document_range)
                    #                   for document_range in long_answer_candidates_range]

                    # if we can generate enough counterfactual and inconsequential documents
                    if counterfactual_documents is not None:
                        # write the generated samples into a new document
                        jsonobj = {
                            "QID": count,
                            "Question": question,
                            "Answers": answers,
                            "URL": json_line['document_url'],
                            "Golden Documents": golden_documents,
                            "Distracting Documents": distracting_document,
                            "Inconsequential Documents": inconsequential_documents,
                            "Low Quality Documents": counterfactual_documents,
                            "Transfer Documents": transfer_documents
                        }
                        json.dump(jsonobj, fileWrite, ensure_ascii=False, indent=4)
                        fileWrite.write(',\n')
                        generated_item += 1
                        print("Question Count:", count)
                        print("origin_question:", question)
                        print("Golden Answer:", answers)
                        print("\n")
            except Exception as e:
                print("Error:", e)
                print("\n")
                time.sleep(20)
# ==================================================== #
            # if count == 30:
            #     break
        fileWrite.write(']\n')
        print('Total Generated Samples Num', generated_item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=str, required=True, help='Number to replace "04" in file paths')
    args = parser.parse_args()

    filePathRead = f'NQ_dataset/v1.0/train/nq-train-{args.num}.jsonl'
    filePathWrite = f'newData/nq-train-{args.num}.json'

    # ==================================================== #
    # filePathRead = '/data1/wangyan/Dataset Process/originData/nq-train-sample.jsonl'
    # filePathWrite = '/data1/wangyan/Dataset Process/newData/nq-train-sample.json'
    data_process(filePathRead, filePathWrite)