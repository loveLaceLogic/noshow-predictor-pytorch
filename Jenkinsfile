pipeline {
    agent any

    stages {
        stage('Verify Files') {
            steps {
                sh 'ls api'
                sh 'ls web'
                sh 'test -f requirements.txt'
            }
        }

        stage('Check Python') {
            steps {
                sh 'python3 --version'
            }
        }

        stage('Simple API Check') {
            steps {
                sh 'python3 -c "from api.main import app; print(\"FastAPI app loaded successfully\")"'
            }
        }
    }
}
