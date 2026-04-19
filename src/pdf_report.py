"""
PDF Report Generator - creates a downloadable credibility report.
"""
from fpdf import FPDF
from datetime import datetime
def clean(text):
    """Remove non-ASCII chars that Helvetica can't render."""
    if not isinstance(text, str):
        text = str(text)
    return text.encode('ascii', 'replace').decode('ascii')
class CredibilityReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(w=self.epw, h=10, text="News Credibility Assessment Report",
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(128, 128, 128)
        self.cell(w=self.epw, h=6,
                  text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(w=self.epw, h=10, text=f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_text_color(33, 37, 41)
        self.set_font("Helvetica", "B", 13)
        self.cell(w=self.epw, h=10, text=title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.set_x(self.l_margin)
        self.multi_cell(w=self.epw, h=6, text=clean(text))
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.set_x(self.l_margin)
        self.multi_cell(w=self.epw, h=6, text=f"  - {clean(text)}")


def generate_pdf_report(report_data):
    """Generate PDF and return as bytes."""
    pdf = CredibilityReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    classification = clean(report_data.get("classification", "UNKNOWN"))
    conf = report_data.get("ml_confidence", 0)
    score = report_data.get("credibility_score", 0)
    pdf.section_title("1. Classification Summary")
    pdf.set_font("Helvetica", "B", 12)
    if classification == "CREDIBLE":
        pdf.set_text_color(39, 174, 96)
    else:
        pdf.set_text_color(231, 76, 60)
    pdf.cell(w=pdf.epw, h=8, text=f"Verdict: {classification}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(w=pdf.epw, h=8,
             text=f"ML Confidence: {conf:.2%}  |  Credibility Score: {score:.2%}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.section_title("2. Article Summary")
    title = clean(report_data.get("article_title", ""))
    if title:
        pdf.set_font("Helvetica", "BI", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=pdf.epw, h=6, text=f"Title: {title}")
        pdf.ln(2)
    pdf.body_text(report_data.get("article_summary", "No summary available."))
    pdf.section_title("3. Credibility Indicators")
    indicators = report_data.get("credibility_indicators", [])
    if indicators:
        for item in indicators:
            pdf.bullet(item)
    else:
        pdf.body_text("No specific credibility indicators found.")
    pdf.ln(4)
    pdf.section_title("4. Risk Factors")
    risks = report_data.get("risk_factors", [])
    if risks:
        for item in risks:
            pdf.bullet(item)
    else:
        pdf.body_text("No significant risk factors detected.")
    pdf.ln(4)
    pdf.section_title("5. Content Analysis")
    sentiment = report_data.get("sentiment", 0)
    style = report_data.get("style_features", [0, 0, 0, 0, 0])
    pdf.body_text(
        f"Sentiment: {sentiment:.3f} ({'Positive' if sentiment >= 0 else 'Negative'})\n"
        f"Exclamations: {int(style[0])}  |  Questions: {int(style[1])}\n"
        f"Caps ratio: {style[2]*100:.1f}%  |  Avg sentence: {style[3]:.1f} words\n"
        f"Word count: {int(style[4])}"
    )
    top_words = report_data.get("top_words", [])
    if top_words:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(w=pdf.epw, h=8, text="Top Influential Words:", new_x="LMARGIN", new_y="NEXT")
        for w, s in top_words[:10]:
            pdf.bullet(f"{w} (score: {s:.4f})")
    pdf.ln(4)
    pdf.section_title("6. Cross-Source Verification")
    pdf.body_text(report_data.get("cross_source_assessment", "Unavailable."))
    articles = report_data.get("verification_articles", [])
    if articles:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(w=pdf.epw, h=8, text="Related Sources:", new_x="LMARGIN", new_y="NEXT")
        for a in articles:
            url = a.get('url', '')
            title = clean(a.get('title', ''))
            source = clean(a.get('source', ''))
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.set_x(pdf.l_margin)
            pdf.write(6, f"  - [{source}] ")
            if url:
                pdf.set_text_color(0, 0, 200) 
                pdf.write(6, title, link=url)
                pdf.set_text_color(60, 60, 60)
            else:
                pdf.write(6, title)
            pdf.ln(6)
    pdf.ln(4)
    pdf.section_title("7. Confidence Assessment")
    pdf.body_text(f"Level: {report_data.get('confidence_level', 'N/A')}")
    explanation = report_data.get("confidence_explanation", "")
    if explanation:
        pdf.body_text(explanation)
    pdf.section_title("8. Recommendation")
    pdf.body_text(report_data.get("recommendation", "No recommendation."))
    pdf.section_title("9. Disclaimer")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 50, 50)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(w=pdf.epw, h=5,
        text=clean(report_data.get("disclaimer", "Automated analysis.")))
    pdf.ln(2)
    pdf.set_text_color(100, 100, 100)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(w=pdf.epw, h=5,
        text="This report was generated by an AI system. It assists but does not replace human judgment.")
    return bytes(pdf.output())