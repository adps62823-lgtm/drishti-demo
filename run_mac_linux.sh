#!/bin/bash
echo ""
echo "  ========================================="
echo "   DRISHTI — AI Vision Assistant"
echo "   by Aditya Pratap Singh"
echo "  ========================================="
echo ""
echo "  Installing / checking dependencies..."
pip install -r requirements.txt -q
echo ""
echo "  Starting DRISHTI server..."
echo "  Open your browser at: http://localhost:5000"
echo ""

# Auto-open browser
sleep 2 && (open http://localhost:5000 2>/dev/null || xdg-open http://localhost:5000 2>/dev/null) &

python app.py
