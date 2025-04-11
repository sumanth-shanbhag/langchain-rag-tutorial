from pdfminer.high_level import extract_text

# Extract text from PDF
pdf_path = "/Users/sumanthshanbhag/Downloads/TwitterSparrow.pdf"
text = extract_text(pdf_path)

# Save as Markdown
md_path = "/Users/sumanthshanbhag/GettingStarted/sumath-git-repo/experimental-1/langchain-rag-tutorial/data/books/sparrow.md"
with open(md_path, "w") as f:
    f.write("# Converted from PDF\n\n")
    f.write(text)