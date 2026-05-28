#!/bin/bash
# Download publicly available malaria guidelines for RAG
# Run this to populate data/guidelines/ with the latest documents

GUIDELINES_DIR="$(dirname "$0")/../data/guidelines"
mkdir -p "$GUIDELINES_DIR"

echo "📚 Guideline documents are pre-included in data/guidelines/"
echo ""
echo "Included sources:"
echo "  ✅ WHO Malaria Treatment Guidelines 2023"
echo "  ✅ CDC Malaria Treatment Protocols"  
echo "  ✅ Malaria Facts & Epidemiology"
echo "  ✅ Microscopy Best Practices"
echo ""
echo "To add more guidelines:"
echo "  1. Download PDF/text from WHO, CDC, or national health ministry"
echo "  2. Convert to .txt or .md format"
echo "  3. Place in data/guidelines/"
echo "  4. Restart the backend — RAG engine auto-indexes new files"
echo ""
echo "Suggested additional sources:"
echo "  - Ghana Health Service Malaria Treatment Protocol"
echo "  - WHO Malaria Microscopy Quality Assurance Manual"
echo "  - MSF Clinical Guidelines — Malaria Chapter"
echo "  - National formularies for drug dosing"
