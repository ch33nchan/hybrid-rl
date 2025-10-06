#!/bin/bash
# Script to compile the LaTeX paper

echo "=========================================="
echo "Compiling LaTeX Paper"
echo "=========================================="

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install a LaTeX distribution (e.g., MacTeX)"
    exit 1
fi

# Compile the paper (run twice for references)
echo ""
echo "First pass..."
pdflatex -interaction=nonstopmode paper.tex

echo ""
echo "Second pass (for references)..."
pdflatex -interaction=nonstopmode paper.tex

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f paper.aux paper.log paper.out paper.bbl paper.blg

echo ""
echo "=========================================="
echo "✓ Compilation complete!"
echo "Output: paper.pdf"
echo "=========================================="

# Open the PDF if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    read -p "Open PDF? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open paper.pdf
    fi
fi
