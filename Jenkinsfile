// ─────────────────────────────────────────────────────────────────────────────
// Jenkinsfile — ADITI / SNN Leaf Guardian CI-CD Pipeline
//
// Triggered on every push to the main branch.
// Stages:
//   1. Checkout        — clone repo
//   2. Build Backend   — docker build backend image
//   3. Build Frontend  — docker build frontend image (npm build inside Docker)
//   4. Test Backend    — smoke-test the Flask /api/health endpoint
//   5. Push Images     — push both images to Docker Hub (tagged :latest + :<BUILD>)
//   6. Deploy          — docker-compose pull + up -d on the target host
//
// Jenkins Setup Required:
//   • Credential ID "dockerhub-creds" — Docker Hub username + password
//   • (Optional) Credential ID "deploy-host-ssh" — SSH key to deployment server
//   • Docker + docker-compose installed on the Jenkins agent
// ─────────────────────────────────────────────────────────────────────────────

pipeline {
    agent any

    environment {
        DOCKER_HUB_USER   = "shubhamp1028"
        BACKEND_IMAGE     = "${DOCKER_HUB_USER}/aditi-backend"
        FRONTEND_IMAGE    = "${DOCKER_HUB_USER}/aditi-frontend"
        BUILD_TAG         = "${env.BUILD_NUMBER}"
        COMPOSE_FILE      = "docker-compose.yml"
    }

    options {
        // Keep last 10 builds
        buildDiscarder(logRotator(numToKeepStr: '10'))
        // Fail if the pipeline takes more than 30 minutes
        timeout(time: 30, unit: 'MINUTES')
        // Do not run concurrent builds on the same branch
        disableConcurrentBuilds()
    }

    triggers {
        // Poll SCM every minute — or set up a GitHub webhook for push events
        pollSCM('* * * * *')
    }

    stages {

        // ── Stage 1 ─────────────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                echo "Checking out source code..."
                checkout scm
            }
        }

        // ── Stage 2 ─────────────────────────────────────────────────────────
        stage('Build Backend Image') {
            steps {
                echo "Building backend Docker image: ${BACKEND_IMAGE}:${BUILD_TAG}"
                sh """
                    docker build \\
                        --tag ${BACKEND_IMAGE}:${BUILD_TAG} \\
                        --tag ${BACKEND_IMAGE}:latest \\
                        --file backend/Dockerfile \\
                        --build-arg BUILDKIT_INLINE_CACHE=1 \\
                        --cache-from ${BACKEND_IMAGE}:latest \\
                        ./backend
                """
            }
        }

        // ── Stage 3 ─────────────────────────────────────────────────────────
        stage('Build Frontend Image') {
            steps {
                echo "Building frontend Docker image: ${FRONTEND_IMAGE}:${BUILD_TAG}"
                sh """
                    docker build \\
                        --tag ${FRONTEND_IMAGE}:${BUILD_TAG} \\
                        --tag ${FRONTEND_IMAGE}:latest \\
                        --file frontend/Dockerfile \\
                        --build-arg BUILDKIT_INLINE_CACHE=1 \\
                        --cache-from ${FRONTEND_IMAGE}:latest \\
                        ./frontend
                """
            }
        }

        // ── Stage 4 ─────────────────────────────────────────────────────────
        stage('Smoke-Test Backend') {
            steps {
                echo "Running backend smoke test..."
                sh """
                    # Start backend container temporarily (model mounted from workspace)
                    docker run -d --name aditi-test-backend \\
                        -p 5001:5000 \\
                        -v \$(pwd)/regular_spiking_model.pt:/app/regular_spiking_model.pt:ro \\
                        ${BACKEND_IMAGE}:${BUILD_TAG}

                    # Wait for readiness (up to 45 seconds)
                    for i in \$(seq 1 45); do
                        STATUS=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/api/health || echo "000")
                        if [ "\$STATUS" = "200" ]; then
                            echo "Backend healthy after \${i}s"
                            break
                        fi
                        echo "Waiting... (\${i}s) [HTTP \$STATUS]"
                        sleep 1
                    done

                    # Verify response
                    curl -sf http://localhost:5001/api/health | grep '"status":"ok"'
                """
            }
            post {
                always {
                    sh "docker stop aditi-test-backend && docker rm aditi-test-backend || true"
                }
            }
        }

        // ── Stage 5 ─────────────────────────────────────────────────────────
        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh """
                        echo "\$DOCKER_PASS" | docker login -u "\$DOCKER_USER" --password-stdin

                        echo "Pushing backend images..."
                        docker push ${BACKEND_IMAGE}:${BUILD_TAG}
                        docker push ${BACKEND_IMAGE}:latest

                        echo "Pushing frontend images..."
                        docker push ${FRONTEND_IMAGE}:${BUILD_TAG}
                        docker push ${FRONTEND_IMAGE}:latest
                    """
                }
            }
        }

        // ── Stage 6 ─────────────────────────────────────────────────────────
        stage('Deploy') {
            steps {
                echo "Deploying updated containers via docker-compose..."
                sh """
                    # Pull the newly pushed images
                    BACKEND_IMAGE=${BACKEND_IMAGE} \\
                    FRONTEND_IMAGE=${FRONTEND_IMAGE} \\
                    docker-compose -f ${COMPOSE_FILE} pull

                    # Recreate containers that have changed, leave others running
                    BACKEND_IMAGE=${BACKEND_IMAGE} \\
                    FRONTEND_IMAGE=${FRONTEND_IMAGE} \\
                    docker-compose -f ${COMPOSE_FILE} up -d --remove-orphans
                """
            }
        }
    }

    // ── Post-build actions ──────────────────────────────────────────────────
    post {
        always {
            echo "Cleaning up local images to free disk space..."
            sh """
                docker rmi ${BACKEND_IMAGE}:${BUILD_TAG}  || true
                docker rmi ${FRONTEND_IMAGE}:${BUILD_TAG} || true
                docker logout || true
            """
        }
        success {
            echo "✅ Pipeline succeeded — Build #${BUILD_TAG} deployed."
        }
        failure {
            echo "❌ Pipeline FAILED — check logs above."
        }
    }
}
