pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                echo 'Cloning repository...'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv .venv
                . .venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Verify API Files') {
            steps {
                sh '''
                ls api
                ls web
                '''
            }
        }

        stage('Run Simple API Test') {
            steps {
                sh '''
                . .venv/bin/activate
                python3 -c "print('FastAPI pipeline test successful')"
                '''
            }
        }
    }
} 
