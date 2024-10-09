
FROM public.ecr.aws/lambda/python:3.11
# Copy the requirements.txt file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

# Copy the lambda function code
COPY src/* ${LAMBDA_TASK_ROOT}

CMD ["main.lambda_handler"]

