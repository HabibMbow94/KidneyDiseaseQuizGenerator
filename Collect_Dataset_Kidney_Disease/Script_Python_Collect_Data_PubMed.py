# # import requests
# # import pandas as pd
# # import time
# # import xml.etree.ElementTree as ET
# # from urllib.parse import quote_plus

# # class PubMedScraper:
# #     def __init__(self, email="hmbow@aimsammi.org", tool="aka_care_edu"):
# #         self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# #         self.email = email  # Remplacez par votre email
# #         self.tool = tool
# #         self.sleep_time = 0.34  # Pour respecter la limite de 3 requêtes par seconde
    
# #     def search_articles(self, query, retmax=100):
# #         """Cherche des articles sur PubMed avec la requête donnée."""
# #         query_url = f"{self.base_url}esearch.fcgi?db=pubmed&term={quote_plus(query)}&retmax={retmax}&retmode=json&usehistory=y&tool={self.tool}&email={self.email}"
# #         response = requests.get(query_url)
# #         time.sleep(self.sleep_time)
        
# #         if response.status_code == 200:
# #             data = response.json()
# #             pmids = data.get("esearchresult", {}).get("idlist", [])
# #             print(f"🔍 Trouvé {len(pmids)} articles correspondant à la requête.")
# #             return pmids
# #         else:
# #             print(f"❌ Erreur lors de la recherche: {response.status_code}")
# #             return []
    
# #     def fetch_article_details(self, pmid):
# #         """Récupère les détails d'un article à partir de son PMID."""
# #         fetch_url = f"{self.base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml&tool={self.tool}&email={self.email}"
# #         response = requests.get(fetch_url)
# #         time.sleep(self.sleep_time)
        
# #         if response.status_code == 200:
# #             return response.text
# #         else:
# #             print(f"❌ Erreur lors de la récupération de l'article {pmid}: {response.status_code}")
# #             return None
    
# #     def parse_article_xml(self, xml_data):
# #         """Analyse les données XML pour extraire les informations pertinentes."""
# #         if xml_data is None:
# #             return None
        
# #         try:
# #             root = ET.fromstring(xml_data)
            
# #             # Initialiser un dictionnaire pour stocker les données
# #             article_data = {
# #                 'pmid': None,
# #                 'title': None,
# #                 'abstract': None,
# #                 'keywords': [],
# #                 'publication_date': None,
# #                 'authors': [],
# #                 'journal': None,
# #                 'doi': None
# #             }
            
# #             # PMID
# #             pmid_element = root.find(".//PMID")
# #             if pmid_element is not None:
# #                 article_data['pmid'] = pmid_element.text
            
# #             # Titre
# #             title_element = root.find(".//ArticleTitle")
# #             if title_element is not None:
# #                 article_data['title'] = title_element.text
            
# #             # Résumé
# #             abstract_texts = root.findall(".//AbstractText")
# #             if abstract_texts:
# #                 article_data['abstract'] = " ".join([abstract.text for abstract in abstract_texts if abstract.text])
            
# #             # Mots-clés
# #             keyword_elements = root.findall(".//Keyword")
# #             article_data['keywords'] = [keyword.text for keyword in keyword_elements if keyword.text]
            
# #             # Date de publication
# #             pub_date = root.find(".//PubDate")
# #             if pub_date is not None:
# #                 year = pub_date.find("Year")
# #                 month = pub_date.find("Month")
# #                 day = pub_date.find("Day")
                
# #                 date_parts = []
# #                 if year is not None and year.text:
# #                     date_parts.append(year.text)
# #                 if month is not None and month.text:
# #                     date_parts.append(month.text)
# #                 if day is not None and day.text:
# #                     date_parts.append(day.text)
                
# #                 if date_parts:
# #                     article_data['publication_date'] = "-".join(date_parts)
            
# #             # Auteurs
# #             author_elements = root.findall(".//Author")
# #             for author in author_elements:
# #                 last_name = author.find("LastName")
# #                 fore_name = author.find("ForeName")
                
# #                 if last_name is not None and last_name.text:
# #                     author_name = last_name.text
# #                     if fore_name is not None and fore_name.text:
# #                         author_name = fore_name.text + " " + author_name
                    
# #                     article_data['authors'].append(author_name)
            
# #             # Journal
# #             journal_element = root.find(".//Journal/Title")
# #             if journal_element is not None and journal_element.text:
# #                 article_data['journal'] = journal_element.text
            
# #             # DOI
# #             article_id_elements = root.findall(".//ArticleId")
# #             for article_id in article_id_elements:
# #                 if article_id.get("IdType") == "doi" and article_id.text:
# #                     article_data['doi'] = article_id.text
# #                     break
            
# #             return article_data
        
# #         except ET.ParseError as e:
# #             print(f"❌ Erreur lors de l'analyse XML: {e}")
# #             return None
    
# #     def search_and_collect(self, query, max_results=100):
# #         """Recherche et collecte les informations des articles."""
# #         pmids = self.search_articles(query, retmax=max_results)
# #         articles_data = []
        
# #         if not pmids:
# #             print("⚠️ Aucune donnée trouvée, vérifiez votre requête PubMed.")
# #             return pd.DataFrame()
        
# #         for i, pmid in enumerate(pmids):
# #             print(f"📄 Traitement de l'article {i+1}/{len(pmids)} (PMID: {pmid})")
# #             xml_data = self.fetch_article_details(pmid)
# #             article_data = self.parse_article_xml(xml_data)
            
# #             if article_data:
# #                 articles_data.append(article_data)
        
# #         # Créer un DataFrame pandas
# #         df = pd.DataFrame(articles_data)
# #         return df
    
# #     def save_to_csv(self, dataframe, filename="pubmed_articles.csv"):
# #         """Sauvegarde les données dans un fichier CSV"""
# #         if not dataframe.empty:
# #             dataframe.to_csv(filename, index=False, encoding='utf-8')
# #             print(f"✅ Données sauvegardées dans {filename}")
# #         else:
# #             print("⚠️ Aucune donnée à sauvegarder.")


# # # ================== 🌟 Exemple d'utilisation ================== #
# # if __name__ == "__main__":
# #     scraper = PubMedScraper()
    
# #     # 🔍 Requête optimisée pour récupérer des articles pertinents
# #     query = "Chronic Kidney Disease[MeSH Terms] AND (Patient Education OR Health Literacy)[Title/Abstract] AND (Questionnaire OR Survey)[Title/Abstract]"
    
# #     # 📥 Collecte des données
# #     articles_df = scraper.search_and_collect(query, max_results=50)
    
# #     # 💾 Sauvegarde des résultats
# #     scraper.save_to_csv(articles_df, "maladies_renales_education_patients.csv")
    
# #     # 📊 Affichage des données collectées
# #     print("\n📢 Aperçu des données collectées :")
# #     print(articles_df.head())



# import requests
# import pandas as pd
# import time
# import xml.etree.ElementTree as ET
# from urllib.parse import quote_plus
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException

# class MedicalScraper:
#     def __init__(self):
#         # Configuration de Selenium avec ChromeDriver
#         options = webdriver.ChromeOptions()
#         options.add_argument("--headless")  # Exécuter en arrière-plan
#         options.add_argument("--no-sandbox")
#         options.add_argument("--disable-dev-shm-usage")
#         options.add_argument("--disable-notifications")
#         options.add_argument("--window-size=1920,1080")
        
#         self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#         self.wait = WebDriverWait(self.driver, 30)

#         # Base PubMed API
#         self.base_pubmed = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
#         self.email = "hmbow@aimsammi.org"
#         self.tool = "aka_care_quiz"
#         self.sleep_time = 3  # Pause pour éviter les blocages

#     ### 🔹 **1️⃣ Scraper PubMed (API)**
#     def search_pubmed(self, query, retmax=50):
#         """Cherche des articles sur PubMed et récupère leurs PMIDs"""
#         query_url = f"{self.base_pubmed}esearch.fcgi?db=pubmed&term={quote_plus(query)}&retmax={retmax}&retmode=json&tool={self.tool}&email={self.email}"
#         response = requests.get(query_url)
#         self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(self.sleep_time)

#         if response.status_code == 200:
#             data = response.json()
#             pmids = data.get('esearchresult', {}).get('idlist', [])
#             print(f"🔍 {len(pmids)} articles trouvés sur PubMed")
#             return pmids
#         else:
#             print(f"⚠️ Erreur API PubMed (Code {response.status_code})")
#             return []

#     def fetch_pubmed_details(self, pmid):
#         """Récupère les détails d'un article PubMed"""
#         fetch_url = f"{self.base_pubmed}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml&tool={self.tool}&email={self.email}"
#         response = requests.get(fetch_url)
#         self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(self.sleep_time)

#         if response.status_code == 200:
#             return response.text
#         else:
#             print(f"⚠️ Erreur récupération article PubMed {pmid} (Code {response.status_code})")
#             return None

#     def parse_pubmed_xml(self, xml_data):
#         """Analyse XML pour extraire les informations d'un article PubMed"""
#         if xml_data is None:
#             return None

#         try:
#             root = ET.fromstring(xml_data)
#             # Extraction des auteurs
#             authors = []
#             for author in root.findall(".//Author"):
#                 last_name = author.findtext(".//LastName", "")
#                 fore_name = author.findtext(".//ForeName", "")
#                 if last_name and fore_name:
#                     authors.append(f"{last_name} {fore_name}")
            
#             # Extraction des mots-clés (MeSH terms)
#             mesh_terms = [term.text for term in root.findall(".//MeshHeading/DescriptorName") if term.text]
            
#             # Création d'un dictionnaire avec toutes les données
#             return {
#                 'source': "PubMed",
#                 'title': root.findtext(".//ArticleTitle"),
#                 'abstract': " ".join([abstract.text for abstract in root.findall(".//AbstractText") if abstract.text]),
#                 'journal': root.findtext(".//Journal/Title"),
#                 'publication_date': root.findtext(".//PubDate/Year"),
#                 'authors': "; ".join(authors),
#                 'mesh_terms': "; ".join(mesh_terms),
#                 'doi': next((id.text for id in root.findall(".//ArticleId") if id.get("IdType") == "doi"), None),
#                 'pmid': root.findtext('.//PMID'),
#                 'url': f"https://pubmed.ncbi.nlm.nih.gov/{root.findtext('.//PMID')}/"
#             }

#         except ET.ParseError as e:
#             print(f"⚠️ Erreur XML: {e}")
#             return None

#     def collect_pubmed(self, query, max_results=50):
#         """Recherche et collecte les articles PubMed"""
#         pmids = self.search_pubmed(query, retmax=max_results)
#         articles = []

#         for pmid in pmids:
#             xml_data = self.fetch_pubmed_details(pmid)
#             article_data = self.parse_pubmed_xml(xml_data)
#             if article_data:
#                 articles.append(article_data)

#         return pd.DataFrame(articles)

#     ### 🔹 **2️⃣ Scraper INSERM (Selenium)**
#     def collect_inserm(self, query="Maladie rénale chronique"):
#         """Scrape les articles sur INSERM avec Selenium"""
#         search_url = f"https://www.inserm.fr/?s={quote_plus(query)}"
#         print("🔄 Accès à INSERM...")
        
#         try:
#             self.driver.get(search_url)
#             time.sleep(5)  # Attendre que la page charge

#             # Scroller pour charger plus de contenu
#             for _ in range(3):
#                 self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#                 time.sleep(3)

#             print("✅ Page INSERM chargée !")

#             # Attendre la présence d'articles
#             self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
#             articles = []
#             elements = self.driver.find_elements(By.TAG_NAME, "article")

#             for article in elements:
#                 try:
#                     title = article.find_element(By.TAG_NAME, "h2").text
#                     link = article.find_element(By.TAG_NAME, "a").get_attribute("href")
#                     summary = article.find_element(By.TAG_NAME, "p").text if article.find_elements(By.TAG_NAME, "p") else "Résumé non disponible"

#                     articles.append({
#                         "source": "INSERM",
#                         "title": title,
#                         "abstract": summary,
#                         "journal": "INSERM",
#                         "publication_date": "Non spécifiée",
#                         "authors": "Non spécifiés",
#                         "mesh_terms": "Non spécifiés",
#                         "doi": None,
#                         "url": link
#                     })
#                 except Exception as e:
#                     print(f"⚠️ Erreur extraction article INSERM: {e}")
#                     continue

#             print(f"🔍 {len(articles)} articles trouvés sur INSERM")
#             return pd.DataFrame(articles)

#         except TimeoutException:
#             print("⚠️ Timeout lors du chargement des résultats INSERM")
#             return pd.DataFrame()

#     ### 🔹 **3️⃣ Scraper OMS (WHO) avec Selenium**
#     def collect_oms(self, query="rénale"):
#         """Scrape les articles de l'OMS avec Selenium"""
#         search_url = f"https://www.who.int/fr/home/search-results?indexCatalogue=genericsearchindex1&searchQuery={quote_plus(query)}&wordsMode=AnyWord"
#         self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(self.sleep_time)
#         self.driver.get(search_url)
        
#         try:
#             # Attendre que les résultats se chargent
#             self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".sf-search-result")))
            
#             articles = []
#             elements = self.driver.find_elements(By.CSS_SELECTOR, ".sf-search-result")

#             for article in elements:
#                 try:
#                     title = article.find_element(By.TAG_NAME, "h2").text
#                     link = article.find_element(By.TAG_NAME, "a").get_attribute("href")
#                     summary = article.find_element(By.TAG_NAME, "p").text
                    
#                     # Tenter d'extraire une date
#                     try:
#                         date_element = article.find_element(By.CSS_SELECTOR, ".sf-search-result__date")
#                         date = date_element.text
#                     except:
#                         date = "Non spécifiée"

#                     articles.append({
#                         "source": "WHO",
#                         "title": title,
#                         "abstract": summary,
#                         "journal": "WHO",
#                         "publication_date": date,
#                         "authors": "Non spécifiés",
#                         "mesh_terms": "Non spécifiés",
#                         "doi": None,
#                         "url": f"https://www.who.int{link}" if link.startswith("/") else link
#                     })
#                 except Exception as e:
#                     print(f"Erreur lors de l'extraction d'un article WHO: {e}")
#                     continue

#             print(f"🔍 {len(articles)} articles trouvés sur WHO")
#             return pd.DataFrame(articles)
            
#         except TimeoutException:
#             print("⚠️ Timeout lors du chargement des résultats WHO")
#             return pd.DataFrame()

#     ### 🔹 **4️⃣ Scraper MIMIC-IV (PhysioNet)**
#     def collect_mimic(self, query="kidney"):
#         """Recherche des données dans MIMIC-IV sur PhysioNet"""
#         # Note: PhysioNet nécessite une authentification
#         # Cette fonction simule la recherche, dans un vrai scénario il faudrait s'authentifier
#         search_url = f"https://physionet.org/search/{quote_plus(query)}"
#         self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(self.sleep_time)
#         self.driver.get(search_url)
        
#         try:
#             # Attendre que les résultats se chargent
#             self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".search-result")))
            
#             datasets = []
#             elements = self.driver.find_elements(By.CSS_SELECTOR, ".search-result")

#             for dataset in elements:
#                 try:
#                     title = dataset.find_element(By.CSS_SELECTOR, "h2 a").text
#                     link = dataset.find_element(By.CSS_SELECTOR, "h2 a").get_attribute("href")
#                     summary = dataset.find_element(By.CSS_SELECTOR, ".search-description").text
                    
#                     # Extraire la date si disponible
#                     try:
#                         date = dataset.find_element(By.CSS_SELECTOR, ".search-footer").text
#                         # Extraire la date du texte
#                         import re
#                         date_match = re.search(r'\d{4}-\d{2}-\d{2}', date)
#                         date = date_match.group(0) if date_match else "Non spécifiée"
#                     except:
#                         date = "Non spécifiée"

#                     datasets.append({
#                         "source": "MIMIC-IV (PhysioNet)",
#                         "title": title,
#                         "abstract": summary,
#                         "journal": "PhysioNet",
#                         "publication_date": date,
#                         "authors": "Non spécifiés",
#                         "mesh_terms": "Non spécifiés",
#                         "doi": None,
#                         "url": link
#                     })
#                 except Exception as e:
#                     print(f"Erreur lors de l'extraction d'un dataset MIMIC: {e}")
#                     continue

#             print(f"🔍 {len(datasets)} datasets trouvés sur PhysioNet/MIMIC")
#             return pd.DataFrame(datasets)
            
#         except TimeoutException:
#             print("⚠️ Timeout lors du chargement des résultats PhysioNet")
#             return pd.DataFrame()

#     ### 🔹 **5️⃣ Fusionner toutes les sources**
#     def collect_all_sources(self, query, max_pubmed_results=50):
#         """Récupère les articles de PubMed, INSERM, WHO et MIMIC"""
#         # Définir les requêtes spécifiques à chaque langue/source
#         pubmed_query = query
#         inserm_query = query if "chronic kidney disease" not in query.lower() else "Maladie rénale chronique"
#         oms_query = query if "chronic kidney disease" not in query.lower() else "rénale chronique"
#         mimic_query = query if "rénale" not in query.lower() else "kidney disease"
        
#         # Collecter les données de chaque source
#         df_pubmed = self.collect_pubmed(pubmed_query, max_pubmed_results)
#         df_inserm = self.collect_inserm(inserm_query)
#         df_oms = self.collect_oms(oms_query)
#         df_mimic = self.collect_mimic(mimic_query)

#         # Concaténer tous les résultats
#         all_articles = pd.concat([df_pubmed, df_inserm, df_oms, df_mimic], ignore_index=True)
        
#         # Supprimer les doublons potentiels
#         all_articles = all_articles.drop_duplicates(subset=['title', 'source'], keep='first')
        
#         return all_articles

#     def save_to_csv(self, dataframe, filename="articles_medecine.csv"):
#         """Sauvegarde les résultats en CSV"""
#         if not dataframe.empty:
#             dataframe.to_csv(filename, index=False, encoding='utf-8')
#             print(f"✅ Données enregistrées dans {filename}")
#         else:
#             print("⚠️ Aucun article trouvé, fichier non sauvegardé.")
            
#     def save_to_json(self, dataframe, filename="articles_medecine.json"):
#         """Sauvegarde les résultats en JSON"""
#         if not dataframe.empty:
#             dataframe.to_json(filename, orient='records', force_ascii=False, indent=4)
#             print(f"✅ Données enregistrées dans {filename}")
#         else:
#             print("⚠️ Aucun article trouvé, fichier non sauvegardé.")

#     def generate_statistics(self, dataframe):
#         """Génère des statistiques sur les données collectées"""
#         if dataframe.empty:
#             return "Aucune donnée à analyser."
            
#         stats = {
#             "total_articles": len(dataframe),
#             "articles_par_source": dataframe['source'].value_counts().to_dict(),
#             "articles_par_annee": dataframe['publication_date'].value_counts().to_dict(),
#             "top_journals": dataframe['journal'].value_counts().head(5).to_dict() if 'journal' in dataframe.columns else {}
#         }
        
#         # Analyse des termes MeSH si disponibles
#         if 'mesh_terms' in dataframe.columns:
#             all_mesh = []
#             for mesh_list in dataframe['mesh_terms'].dropna():
#                 if isinstance(mesh_list, str):
#                     all_mesh.extend([term.strip() for term in mesh_list.split(';')])
            
#             from collections import Counter
#             mesh_counter = Counter(all_mesh)
#             stats["top_mesh_terms"] = dict(mesh_counter.most_common(10))
            
#         return stats

#     def close_driver(self):
#         """Ferme Selenium"""
#         self.driver.quit()

# # 🎯 **Exécution**
# if __name__ == "__main__":
#     scraper = MedicalScraper()

#     # 🔍 **Requête - options de recherche**
#     queries = [
#         "Maladie rénale chronique", 
#         "Chronic kidney disease",
#         "Insuffisance rénale",
#         "Renal failure",
#         "Dialyse rénale",
#         "Kidney dialysis",
#         "Néphrologie",
#         "Nephrology"
#     ]
    
#     # Créer un DataFrame vide pour stocker tous les résultats
#     all_results = pd.DataFrame()
    
#     # 📥 **Collecte de toutes les sources pour chaque requête**
#     for query in queries:
#         print(f"\n🔍 Recherche pour: {query}")
#         articles_df = scraper.collect_all_sources(query, max_pubmed_results=30)
#         # Ajouter les résultats au DataFrame global
#         all_results = pd.concat([all_results, articles_df], ignore_index=True)
#         # Pause pour éviter de surcharger les serveurs
#         time.sleep(3)
    
#     # Supprimer les doublons du DataFrame global
#     all_results = all_results.drop_duplicates(subset=['title', 'url'], keep='first')
    
#     # 💾 **Sauvegarde en différents formats**
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     scraper.save_to_csv(all_results, f"articles_maladies_renales_{timestamp}.csv")
#     scraper.save_to_json(all_results, f"articles_maladies_renales_{timestamp}.json")
    
#     # 📊 **Générer des statistiques**
#     stats = scraper.generate_statistics(all_results)
#     print("\n📊 Statistiques des données collectées :")
#     import json
#     print(json.dumps(stats, indent=4, ensure_ascii=False))
    
#     # 👀 **Aperçu**
#     print("\n📊 Aperçu des données collectées :")
#     print(all_results.head())
    
#     # 🔄 **Fermer Selenium**
#     scraper.close_driver()



import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from deep_translator import GoogleTranslator

class PubMedScraper:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.sleep_time = 1  # Pause pour éviter les blocages
    
    def search_articles(self, query, retmax=50):
        """Cherche des articles sur PubMed avec une requête."""
        query_url = f"{self.base_url}esearch.fcgi?db=pubmed&term={quote_plus(query)}&retmax={retmax}&retmode=json"
        response = requests.get(query_url)
        time.sleep(self.sleep_time)
        
        if response.status_code == 200:
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            return pmids
        else:
            print(f"❌ Erreur API PubMed (Code {response.status_code})")
            return []
    
    def fetch_article_details(self, pmid):
        """Récupère les détails d'un article via son PMID."""
        fetch_url = f"{self.base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
        response = requests.get(fetch_url)
        time.sleep(self.sleep_time)
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"❌ Erreur récupération article {pmid}")
            return None
    
    def parse_article_xml(self, xml_data):
        """Analyse les données XML pour extraire le titre et le résumé."""
        if xml_data is None:
            return None
        
        try:
            root = ET.fromstring(xml_data)
            title = root.findtext(".//ArticleTitle")
            abstract_texts = root.findall(".//AbstractText")
            abstract = " ".join([abstract.text for abstract in abstract_texts if abstract.text])
            return {"title": title, "abstract": abstract}
        except ET.ParseError as e:
            print(f"❌ Erreur XML: {e}")
            return None
    
    def collect_articles(self, query, max_results=50):
        """Recherche et collecte les articles PubMed."""
        pmids = self.search_articles(query, retmax=max_results)
        articles = []

        for pmid in pmids:
            xml_data = self.fetch_article_details(pmid)
            article_data = self.parse_article_xml(xml_data)
            if article_data:
                articles.append(article_data)

        return pd.DataFrame(articles)

# 🔍 Exécution
scraper = PubMedScraper()
query = "Chronic Kidney Disease[MeSH Terms] OR Renal Insufficiency OR Dialysis OR Nephrology"
df = scraper.collect_articles(query, max_results=50)

# Traduction en français
translator = GoogleTranslator(source="en", target="fr")
df["abstract_fr"] = df["abstract"].apply(lambda x: translator.translate(x) if pd.notna(x) else "")

# Sauvegarde des articles
df.to_csv("articles_maladies_renales.csv", index=False, encoding="utf-8")
print("✅ Articles récupérés et sauvegardés !")