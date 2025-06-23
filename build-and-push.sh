#!/bin/bash

echo "==============================================="
echo "MCP Crawl4AI RAG - Build and Push to Docker Hub"
echo "==============================================="
echo ""
echo "Registry: Docker Hub"
echo ""

echo "Choose build option:"
echo "1. MCP Server only (without Ollama)"
echo "2. MCP Server + Ollama (full stack)"
echo ""
read -p "Enter your choice (1 or 2): " BUILD_CHOICE

case $BUILD_CHOICE in
    1)
        echo "Selected: MCP Server only"
        TARGET_IMAGE="drnit29/mcp-crawl4ai-rag:latest"
        COMPOSE_FILE="docker-compose.yml"
        BUILD_TAG="mcp-only"
        ;;
    2)
        echo "Selected: MCP Server + Ollama"
        TARGET_IMAGE="drnit29/mcp-crawl4ai-rag:with-ollama"
        COMPOSE_FILE="docker-compose.with-ollama.yml"
        BUILD_TAG="with-ollama"
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "Target image: $TARGET_IMAGE"
echo "Using compose file: $COMPOSE_FILE"
echo ""

echo "[1/8] Stopping existing containers..."
docker-compose down
docker-compose -f docker-compose.with-ollama.yml down
docker-compose -f docker-compose.published.yml down
if [ $? -ne 0 ]; then
    echo "Warning: Error stopping containers, continuing..."
fi
echo ""

echo "[2/8] Building MCP server container with custom tag..."
echo "This may take several minutes due to dependencies..."
docker build -t $TARGET_IMAGE --build-arg PORT=8051 .
if [ $? -ne 0 ]; then
    echo "Error: Failed to build MCP server container"
    read -p "Press any key to continue..."
    exit 1
fi
echo "Build completed successfully!"
echo ""

echo "[3/8] Verifying built image..."
docker images | grep "drnit29/mcp-crawl4ai-rag"
if [ $? -ne 0 ]; then
    echo "Error: Built image not found"
    read -p "Press any key to continue..."
    exit 1
fi
echo "Image verification successful!"
echo ""

echo "[4/8] Starting container with built image for testing..."
echo "Stopping any existing containers first..."
docker-compose down >/dev/null 2>&1
docker-compose -f docker-compose.with-ollama.yml down >/dev/null 2>&1
docker-compose -f docker-compose.published.yml down >/dev/null 2>&1

echo ""
echo "Starting services using new built image..."
docker-compose -f $COMPOSE_FILE up -d
if [ $? -ne 0 ]; then
    echo "Error: Failed to start services with built image"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if image was built: docker images | grep drnit29/mcp-crawl4ai-rag"
    echo "2. Verify network connectivity: docker network ls"
    echo "3. Check logs: docker-compose -f $COMPOSE_FILE logs"
    echo ""
    read -p "Press any key to continue..."
    exit 1
fi

echo "Waiting for services to initialize..."
if [ "$BUILD_CHOICE" == "2" ]; then
    echo "Waiting longer for Ollama to initialize..."
    sleep 30
else
    sleep 10
fi

echo "Testing service health..."
sleep 5
curl -f http://localhost:8051/health >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ MCP Server is healthy and responding!"
    echo "✅ Built image is working correctly"
    echo ""
else
    echo "⚠️  MCP Server is still starting up"
    echo "Waiting a bit more..."
    sleep 10
    curl -f http://localhost:8051/health >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ MCP Server is now healthy!"
    else
        echo "❌ MCP Server health check failed"
        echo "Container logs:"
        docker-compose -f $COMPOSE_FILE logs --tail=20 mcp-server
        read -p "Press any key to continue..."
        exit 1
    fi
fi

echo "Services Status:"
docker-compose -f $COMPOSE_FILE ps
echo ""

echo "[5/8] Docker Hub login..."
echo ""
echo "Please enter your Docker Hub credentials:"
docker login
if [ $? -ne 0 ]; then
    echo "Error: Docker Hub login failed"
    echo ""
    echo "Make sure you have a Docker Hub account and correct credentials"
    echo "Visit: https://hub.docker.com/signup"
    read -p "Press any key to continue..."
    exit 1
fi
echo "Docker Hub login successful!"
echo ""

echo "[6/8] Pushing image to Docker Hub..."
echo "This may take several minutes depending on your upload speed..."
echo ""
docker push $TARGET_IMAGE
if [ $? -ne 0 ]; then
    echo "Error: Failed to push image to Docker Hub"
    echo ""
    echo "Possible causes:"
    echo "1. Network connectivity issues"
    echo "2. Docker Hub authentication expired"
    echo "3. Repository permissions"
    echo ""
    read -p "Press any key to continue..."
    exit 1
fi
echo "Image pushed successfully to Docker Hub!"
echo ""

echo "[7/8] Creating additional tags..."
if [ "$BUILD_CHOICE" == "1" ]; then
    echo "Creating 'latest' and 'v1.0' tags..."
    docker tag $TARGET_IMAGE drnit29/mcp-crawl4ai-rag:v1.0
    docker push drnit29/mcp-crawl4ai-rag:v1.0
else
    echo "Creating 'with-ollama-v1.0' tag..."
    docker tag $TARGET_IMAGE drnit29/mcp-crawl4ai-rag:with-ollama-v1.0
    docker push drnit29/mcp-crawl4ai-rag:with-ollama-v1.0
fi
if [ $? -ne 0 ]; then
    echo "Warning: Failed to push additional tag, but main image was successful"
fi
echo ""

echo "[8/8] Finalizing deployment..."
echo "Services are already running with the newly built and pushed image."
echo "Verifying final status..."
echo ""

echo "==============================================="
echo "Build and Push completed successfully!"
echo "==============================================="
echo ""
echo "Docker Hub Repository: https://hub.docker.com/r/drnit29/mcp-crawl4ai-rag"
echo "Image Tags:"
echo "- drnit29/mcp-crawl4ai-rag:latest"
echo "- drnit29/mcp-crawl4ai-rag:v1.0"
echo ""

echo "Final Services Status:"
docker-compose -f $COMPOSE_FILE ps
echo ""

echo "Final health verification..."
curl -f http://localhost:8051/health >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ MCP Server is healthy and ready for use!"
    echo "✅ Image successfully built, tested, pushed, and deployed!"
    curl -f http://localhost:8051/health?format=json 2>/dev/null
    echo ""
else
    echo "⚠️  Note: Service may still be initializing"
    echo "    Status check: curl http://localhost:8051/health"
fi
echo ""

echo "Available endpoints:"
echo "- MCP Server: http://localhost:8051"
echo "- Health Check: http://localhost:8051/health"
echo "- Health Check (JSON): http://localhost:8051/health?format=json"
if [ "$BUILD_CHOICE" == "2" ]; then
    echo "- Ollama: http://localhost:11434"
fi
echo ""

echo "Management commands:"
echo "- View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "- Stop services: docker-compose -f $COMPOSE_FILE down"
echo "- Restart services: docker-compose -f $COMPOSE_FILE restart"
echo ""

echo "Usage for others:"
if [ "$BUILD_CHOICE" == "1" ]; then
    echo "docker pull drnit29/mcp-crawl4ai-rag:latest"
    echo "docker run -p 8051:8051 --env-file .env drnit29/mcp-crawl4ai-rag:latest"
else
    echo "docker pull drnit29/mcp-crawl4ai-rag:with-ollama"
    echo "docker-compose -f docker-compose.published.yml up -d"
fi
echo ""

echo "Services are now running with the published image! 🚀"
read -p "Press any key to continue..."