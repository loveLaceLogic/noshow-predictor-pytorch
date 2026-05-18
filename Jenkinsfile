pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                sh 'python3 -m venv .venv'
                sh '. .venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Run API') {
            steps {
                sh '. .venv/bin/activate && python -m uvicorn api.main:app --reload'
            }
        }
    }
}
