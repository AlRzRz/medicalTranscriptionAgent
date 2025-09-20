import time
import pandas as pd
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# GPT-like fluid text streaming into the terminal
def streamText(text):

    for char in text:
        time.sleep(0.015)
        print(char, end='', flush=True)

    print()


def main():
    df = pd.read_csv("transcriptions.csv")
    df.head()

    load_dotenv()
    client = OpenAI(
        api_key=os.getenv('OPENAPI_KEY'))


    medSpecialtyList = df['medical_specialty'].to_list()
    transcriptionList = df['transcription'].to_list()


    mainTools = [
    {
        "type": "function",
        'function': {
            "name": "get_main_medical_info",
            "description": "Extract the 'age' and 'recommended_treatment' from each transcription",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "string",
                        "description": "The patient age extracted from the transcription.",
                    },
                    "recommended_treatment": {
                        "type": "string",
                        "description": "The recommended treatment extracted from each transcription.",
                    },
                },
            },
        },
    },
    ]

    transcriptionContext = '||'.join(transcriptionList)
    print(transcriptionContext)

    messages=[
        {'role': 'system', 'content': 'You are a medical transcription agent. You extract medical information from these transcripts and automate the matching with the appropriate ICD-10 codes. ICD-10 codes are a standardized system used worldwide for diagnosing and billing purposes, such as insurance claims processing.'}
    ]

    messages.append({
        'role': 'user',
        'content': f"""
        These are medical transcriptions separated with a '||' in proper order. Apply the appropriate tools to complete your function please.
        
        {transcriptionContext}"""
    })

    response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=messages,
    tools=mainTools
    
)

    print(response)

    print(response.choices[0].message.tool_calls)

    listOfMainInfo = response.choices[0].message.tool_calls

    finalDf = pd.DataFrame()
    finalDf['medical_specialty'] = df['medical_specialty']
    print(finalDf.head())

    mainInfoListOfDicts = [json.loads(resp.function.arguments) for resp in listOfMainInfo]
    print(mainInfoListOfDicts)

    intermediaryDf = pd.DataFrame(mainInfoListOfDicts)
    print(intermediaryDf)

    finalDf['age'] = intermediaryDf['age']
    finalDf['recommended_treatment'] = intermediaryDf['recommended_treatment']

    lstOfRecommendedTreatments = finalDf['recommended_treatment'].values.tolist()

    ICDTools = [
    {
        "type": "function",
        'function': {
            "name": "get_icd_code",
            "description": "Match each recommended treatment with the corresponding International Classification of Diseases (ICD) code",
            "parameters": {
                "type": "object",
                "properties": {
                    "icd_code": {
                        "type": "string",
                        "description": "An ICD-10 code which is a standardized system used worldwide for diagnosing and billing purposes, such as insurance claims processing.",
                    },
                },
            },
        },
    },
    ]

    ICDmessages=[
            {'role': 'system', 'content': 'You are a medical transcription agent. You match recommended medical treatments from these patient\'s recommended treatments match them with the appropriate ICD-10 codes. ICD-10 codes are a standardized system used worldwide for diagnosing and billing purposes, such as insurance claims processing.'}
        ]

    ICDmessages.append({
        'role': 'user',
        'content': f"""
        These are recommended medical treatments separated with a '||' in proper order. Apply the appropriate tools to complete your function please.
        
        {lstOfRecommendedTreatments}"""
    })

    ICDresponse = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=ICDmessages,
    tools=ICDTools
    
    )

    print(ICDresponse.choices[0].message.tool_calls)

    icds = ICDresponse.choices[0].message.tool_calls

    lstOfIcds = [json.loads(icd.function.arguments) for icd in icds]
    print(lstOfIcds)

    intermDf = pd.DataFrame(lstOfIcds)

    finalDf['icd_code'] = intermDf['icd_code']

    df_structured = finalDf
    print(df_structured)

main()