pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv .venv
                . .venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Verify Project Structure') {
            steps {
                sh '''
                ls api
                ls web
                test -f requirements.txt
                '''
            }
        }

        stage('Test FastAPI Import') {
            steps {
                sh '''
                . .venv/bin/activate
                python3 -c "from api.main import app; print('FastAPI app loaded successfully')"
                '''
            }
        }
    }
}
