import azure.functions as func
import logging
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = func.FunctionApp()

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


@app.route(route="HealthCheck", auth_level=func.AuthLevel.ANONYMOUS)
def HealthCheck(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("Healthy", status_code=200)

@app.route(route="DiseaseDetectorTrigger", auth_level=func.AuthLevel.ANONYMOUS)
def DiseaseDetectorTrigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()

    description = req_body.get('description')
    if not description:
        return func.HttpResponse(
            "No description is provided to the ner model",
            status_code=400
        )

    response = pipe(description)
    first_sign_symptom = None
    first_biological_structures = None

    for entity in response:
        if entity['entity_group'] == 'Sign_symptom' and first_sign_symptom is None:
            first_sign_symptom = entity['word']
        if entity['entity_group'] == 'Biological_structure' and first_biological_structures is None:
            first_biological_structures = entity['word']

    if first_biological_structures and first_sign_symptom:
        result = first_biological_structures + ' ' + first_sign_symptom
    else:
        result = "Not enough data to form the result"
    return func.HttpResponse(result.title(), status_code=200)


