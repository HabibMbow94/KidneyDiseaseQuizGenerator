import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class MedicalScraper:
    def __init__(self):
        # Configuration de Selenium avec ChromeDriver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Exécuter en arrière-plan
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # Base PubMed API
        self.base_pubmed = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = "hmbow@aimsammi.org"
        self.tool = "aka_care_quiz"
        self.sleep_time = 1  # Pause pour éviter les blocages

    ### 🔹 **1️⃣ Scraper PubMed (API)**
    def search_pubmed(self, query, retmax=50):
        """Cherche des articles sur PubMed et récupère leurs PMIDs"""
        query_url = f"{self.base_pubmed}esearch.fcgi?db=pubmed&term={quote_plus(query)}&retmax={retmax}&retmode=json"
        response = requests.get(query_url)
        time.sleep(self.sleep_time)

        if response.status_code == 200:
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            print(f"🔍 {len(pmids)} articles trouvés sur PubMed")
            return pmids
        else:
            print(f"⚠️ Erreur API PubMed (Code {response.status_code})")
            return []

    def fetch_pubmed_details(self, pmid):
        """Récupère les détails d'un article PubMed"""
        fetch_url = f"{self.base_pubmed}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
        response = requests.get(fetch_url)
        time.sleep(self.sleep_time)

        if response.status_code == 200:
            return response.text
        else:
            print(f"⚠️ Erreur récupération article PubMed {pmid} (Code {response.status_code})")
            return None

    def parse_pubmed_xml(self, xml_data):
        """Analyse XML pour extraire les informations d'un article PubMed"""
        if xml_data is None:
            return None

        try:
            root = ET.fromstring(xml_data)
            return {
                'source': "PubMed",
                'title': root.findtext(".//ArticleTitle"),
                'abstract': " ".join([abstract.text for abstract in root.findall(".//AbstractText") if abstract.text]),
                'journal': root.findtext(".//Journal/Title"),
                'publication_date': root.findtext(".//PubDate/Year"),
                'doi': next((id.text for id in root.findall(".//ArticleId") if id.get("IdType") == "doi"), None),
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{root.findtext('.//PMID')}/"
            }

        except ET.ParseError as e:
            print(f"⚠️ Erreur XML: {e}")
            return None

    def collect_pubmed(self, query, max_results=50):
        """Recherche et collecte les articles PubMed"""
        pmids = self.search_pubmed(query, retmax=max_results)
        articles = []

        for pmid in pmids:
            xml_data = self.fetch_pubmed_details(pmid)
            article_data = self.parse_pubmed_xml(xml_data)
            if article_data:
                articles.append(article_data)

        return pd.DataFrame(articles)

    ### 🔹 **2️⃣ Scraper INSERM (Selenium)**
    def collect_inserm(self, query="Maladie rénale chronique"):
        """Scrape les articles sur INSERM avec Selenium"""
        search_url = f"https://www.inserm.fr/?s={quote_plus(query)}"
        self.driver.get(search_url)
        time.sleep(self.sleep_time)

        articles = []
        elements = self.driver.find_elements(By.CSS_SELECTOR, ".list-articles .article-list-item")

        for article in elements:
            try:
                title = article.find_element(By.TAG_NAME, "h2").text
                link = article.find_element(By.TAG_NAME, "a").get_attribute("href")
                summary = article.find_element(By.TAG_NAME, "p").text

                articles.append({
                    "source": "INSERM",
                    "title": title,
                    "abstract": summary,
                    "journal": "INSERM",
                    "publication_date": "Non spécifiée",
                    "doi": None,
                    "url": link
                })
            except:
                continue

        print(f"🔍 {len(articles)} articles trouvés sur INSERM")
        return pd.DataFrame(articles)

    ### 🔹 **3️⃣ Scraper OMS (WHO) avec Selenium**
    def collect_oms(self, query="rénale"):
        """Scrape les articles de l'OMS avec Selenium"""
        search_url = f"https://www.who.int/fr/home/search-results?indexCatalogue=genericsearchindex1&searchQuery={quote_plus(query)}&wordsMode=AnyWord"
        self.driver.get(search_url)
        time.sleep(self.sleep_time)

        articles = []
        elements = self.driver.find_elements(By.CSS_SELECTOR, ".sf-search-result")

        for article in elements:
            try:
                title = article.find_element(By.TAG_NAME, "h2").text
                link = article.find_element(By.TAG_NAME, "a").get_attribute("href")
                summary = article.find_element(By.TAG_NAME, "p").text

                articles.append({
                    "source": "WHO",
                    "title": title,
                    "abstract": summary,
                    "journal": "WHO",
                    "publication_date": "Non spécifiée",
                    "doi": None,
                    "url": f"https://www.who.int{link}" if link.startswith("/") else link
                })
            except:
                continue

        print(f"🔍 {len(articles)} articles trouvés sur WHO")
        return pd.DataFrame(articles)

    ### 🔹 **4️⃣ Fusionner toutes les sources**
    def collect_all_sources(self, query, max_results=50):
        """Récupère les articles de PubMed, INSERM et WHO"""
        df_pubmed = self.collect_pubmed(query, max_results)
        df_inserm = self.collect_inserm(query)
        df_oms = self.collect_oms(query)

        all_articles = pd.concat([df_pubmed, df_inserm, df_oms], ignore_index=True)
        return all_articles

    def save_to_csv(self, dataframe, filename="articles_medecine.csv"):
        """Sauvegarde les résultats en CSV"""
        if not dataframe.empty:
            dataframe.to_csv(filename, index=False, encoding='utf-8')
            print(f"✅ Données enregistrées dans {filename}")
        else:
            print("⚠️ Aucun article trouvé, fichier non sauvegardé.")

    def close_driver(self):
        """Ferme Selenium"""
        self.driver.quit()

# 🎯 **Exécution**
if __name__ == "__main__":
    scraper = MedicalScraper()

    # 🔍 **Requête**
    query = "Maladie rénale chronique"

    # 📥 **Collecte de toutes les sources**
    articles_df = scraper.collect_all_sources(query, max_results=50)

    # 💾 **Sauvegarde**
    scraper.save_to_csv(articles_df, "articles_maladies_renales.csv")

    # 👀 **Aperçu**
    print("\n📊 Aperçu des données collectées :")
    print(articles_df.head())

    # 🔄 **Fermer Selenium**
    scraper.close_driver()