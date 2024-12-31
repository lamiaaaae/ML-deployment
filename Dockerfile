FROM public.ecr.aws/lambda/python:3.10

# Copier le modèle pré-entraîné et le fichier de dépendances
COPY model.pkl . 
COPY requirements.txt . 

# Installer les dépendances
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copier le fichier de fonction FastAPI
COPY app.py ${LAMBDA_TASK_ROOT}

# Définir le point d'entrée pour la fonction Lambda
CMD ["app.handler"]
