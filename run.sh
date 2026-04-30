#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AgriPredict...${NC}"

# Start Flask backend on port 5000
echo -e "${BLUE}[Backend]${NC} Starting Flask on http://localhost:5000"
python app.py &
FLASK_PID=$!

# Start frontend static server on port 8080
echo -e "${BLUE}[Frontend]${NC} Starting frontend on http://localhost:8080"
python -m http.server 8080 &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}✓ Both servers running!${NC}"
echo -e "  Frontend → http://localhost:8080"
echo -e "  Backend  → http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers."

# Kill both on Ctrl+C
trap "echo ''; echo 'Shutting down...'; kill $FLASK_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

wait