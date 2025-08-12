import logging
import base64
import azure.functions as func
import fitz
import requests
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from msal import ConfidentialClientApplication
import os
import tempfile
import json
import datetime
from azure.storage.blob import BlobServiceClient
from PIL import Image
from utils.shp_access import (
    get_access_token as shp_get_access_token,
    get_site_id as shp_get_site_id,
    get_drive_id as shp_get_drive_id,
    list_drive_folder as shp_list_drive_folder
)
from utils.image_utils import detect_first_page_table, detect_and_crop_tables
import time
import cv2
import numpy as np
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential as DIAzureKeyCredential
import uuid
import re

# --- Configuración de Azure Cognitive Search ---
admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")

settings_path = os.path.join(os.path.dirname(__file__), "local.settings.json")
if (not all([admin_key, index_name, endpoint])) and os.path.exists(settings_path):
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
        values = settings.get("Values", {})
        admin_key = admin_key or values.get("AZURE_SEARCH_ADMIN_KEY")
        index_name = index_name or values.get("AZURE_SEARCH_INDEX_NAME")
        endpoint = endpoint or values.get("AZURE_SEARCH_ENDPOINT")

if not all([admin_key, index_name, endpoint]):
    raise EnvironmentError("AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME y AZURE_SEARCH_ENDPOINT deben estar definidos y no vacíos.")

search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(str(admin_key)))

# --- Configuración de Azure Blob Storage ---
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

# Leer credenciales de local.settings.json si no están en el entorno
if not all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_STORAGE_CONTAINER]):
    settings_path = os.path.join(os.path.dirname(__file__), "local.settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
            values = settings.get("Values", {})
            AZURE_STORAGE_ACCOUNT = AZURE_STORAGE_ACCOUNT or values.get("AZURE_STORAGE_ACCOUNT")
            AZURE_STORAGE_KEY = AZURE_STORAGE_KEY or values.get("AZURE_STORAGE_KEY")
            AZURE_STORAGE_CONTAINER = AZURE_STORAGE_CONTAINER or values.get("AZURE_STORAGE_CONTAINER")

if not all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_STORAGE_CONTAINER]):
    raise EnvironmentError("AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY y AZURE_STORAGE_CONTAINER deben estar definidos y no vacíos.")

blob_service_client = BlobServiceClient(
    f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=AZURE_STORAGE_KEY
)
container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)
try:
    container_client.create_container()
except Exception:
    pass  # Ya existe

# --- Funciones auxiliares ---
def generate_embeddings(chunks):
    """
    Genera embeddings usando Azure OpenAI con el cliente moderno openai v1.x
    """
    embeddings = []
    
    try:
        # Configuración usando las variables de entorno
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("EMBEDDING_DEPLOYMENT")
        
        if not api_key:
            logging.error("Azure OpenAI API key not found")
            return [[] for _ in chunks]  # Retornar embeddings vacíos
        
        # Configurar cliente OpenAI para Azure usando el formato correcto para embeddings
        client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=endpoint,
            api_key=api_key
        )
        
        logging.info(f"Generating embeddings with deployment: {deployment}")
        
        # Generar embeddings para todos los chunks de una vez
        response = client.embeddings.create(
            input=chunks,
            model=deployment
        )
        
        # Extraer embeddings de la respuesta
        for item in response.data:
            embeddings.append(item.embedding)
            
        logging.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return [[] for _ in chunks]  # Retornar embeddings vacíos en caso de error

def blob_exists(blob_name):
    """
    Verifica si un blob existe en el contenedor de Azure Blob Storage.
    """
    try:
        container_client.get_blob_client(blob_name).get_blob_properties()
        return True
    except Exception:
        return False

def read_blob_json(blob_name):
    """
    Lee y decodifica un blob JSON desde Azure Blob Storage.
    """
    if not blob_exists(blob_name):
        return None
    blob_client = container_client.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as e:
        logging.warning(f"El blob '{blob_name}' no contiene un JSON válido: {e}")
        return None

def write_blob_json(blob_name, data):
    """
    Escribe un objeto como JSON en un blob de Azure Blob Storage.
    """
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(json.dumps(data), overwrite=True)

def write_blob_file(blob_name, data):
    """
    Escribe datos binarios en un blob de Azure Blob Storage.
    """
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)

def load_vision_agent_prompt():
    """
    Carga el prompt del vision agent desde el archivo vision_agent_prompt.txt
    """
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "vision_agent_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error al cargar el prompt del vision agent: {e}")
        return ""

def call_vision_agent(image_path, prompt, max_retries=3):
    """
    Llama al vision agent con una imagen y un prompt con reintentos en caso de errores JSON
    """
    for attempt in range(max_retries):
        try:
            # Configuración del endpoint de vision agent
            vision_endpoint = os.getenv("VISION_AGENT_ENDPOINT")
            vision_deployment = os.getenv("VISION_AGENT_DEPLOYMENT")
            api_key = os.getenv("API_KEY")
            # Use a stable API version for vision models
            api_version = "2024-02-01"
            
            # Convertir imagen a base64
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key
            }
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 10000,
                "temperature": 0.1
            }
            
            url = f"{vision_endpoint}/openai/deployments/{vision_deployment}/chat/completions?api-version={api_version}"
            
            if attempt == 0:  # Log details only on first attempt
                logging.info(f"Vision API call details:")
                logging.info(f"  URL: {url}")
                logging.info(f"  Deployment: {vision_deployment}")
                logging.info(f"  API Version: {api_version}")
            else:
                logging.info(f"Vision API retry attempt {attempt + 1}/{max_retries}")
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                logging.error(f"Vision API error {response.status_code}: {response.text}")
                if attempt == max_retries - 1:  # Last attempt
                    response.raise_for_status()
                else:
                    continue  # Retry on next iteration
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Test if the response can be parsed as JSON
            try:
                cleaned_response = clean_json_response(content)
                json.loads(cleaned_response)  # Test parse
                
                # If we get here, the JSON is valid
                logging.info(f"=== VISION AGENT RESPONSE (Attempt {attempt + 1}) ===")
                logging.info(f"Response: {content}")
                logging.info(f"=== END VISION AGENT RESPONSE ===")
                
                return content
                
            except json.JSONDecodeError as json_error:
                logging.warning(f"Attempt {attempt + 1}/{max_retries} - Invalid JSON response: {json_error}")
                logging.warning(f"Response preview: {content[:200]}...")
                
                if attempt == max_retries - 1:  # Last attempt
                    logging.error(f"All {max_retries} attempts failed due to JSON parsing errors")
                    return None
                else:
                    logging.info(f"Retrying vision agent call...")
                    continue
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries} - Error calling vision agent: {e}")
            if attempt == max_retries - 1:  # Last attempt
                return None
            else:
                logging.info(f"Retrying due to error...")
                continue
    
    return None

def process_with_document_intelligence(image_path):
    """
    Procesa una imagen con Azure Document Intelligence
    """
    try:
        endpoint = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
        
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=DIAzureKeyCredential(key))
        
        with open(image_path, "rb") as f:
            poller = client.begin_analyze_document(
                "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
            )
        
        result = poller.result()
        
        # Extraer texto del resultado
        content = ""
        if result.content:
            content = result.content
        
        logging.info(f"*** DOCUMENT INTELLIGENCE PROCESSED ***")
        logging.info(f"Extracted content length: {len(content)}")
        logging.info(f"*** END DOCUMENT INTELLIGENCE ***")
        
        return content
    except Exception as e:
        logging.error(f"Error al procesar con Document Intelligence: {e}")
        return ""

def clean_json_response(response_text):
    """
    Limpia la respuesta del vision agent removiendo markdown formatting
    """
    if not response_text:
        return response_text
    
    # Remove markdown code block formatting
    cleaned = response_text.strip()
    
    # Remove ```json and ``` markers
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]  # Remove ```json
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]   # Remove ```
    
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]  # Remove trailing ```
    
    return cleaned.strip()

def extract_date_from_text(text):
    """
    Extrae fecha en formato MM/DD/YYYY del texto usando regex
    """
    try:
        # Patrón para fecha MM/DD/YYYY
        date_pattern = r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'
        match = re.search(date_pattern, text)
        
        if match:
            month, day, year = match.groups()
            date_str = f"{month.zfill(2)}/{day.zfill(2)}/{year}"
            # Convertir a DateTimeOffset
            date_obj = datetime.datetime.strptime(date_str, "%m/%d/%Y")
            return date_obj.isoformat() + "Z"
        return None
    except Exception as e:
        logging.error(f"Error al extraer fecha: {e}")
        return None

def pdf_to_images(pdf_path, max_pages=3):
    """
    Convierte las primeras páginas de un PDF a imágenes usando EXACTAMENTE la misma lógica que opencv-tests.py
    """
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(min(max_pages, len(doc))):
            page = doc.load_page(page_num)
            # Usar exactamente la misma configuración que opencv-tests.py
            pix = page.get_pixmap(dpi=300)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height,
                pix.width,
                pix.n
            )
            
            # Guardar imagen temporal
            temp_path = tempfile.mktemp(suffix=f'_page_{page_num}.png')
            cv2.imwrite(temp_path, img_data)
            images.append(temp_path)
        doc.close()
        return images
    except Exception as e:
        logging.error(f"Error al convertir PDF a imágenes: {e}")
        return []

def main(mytimer: func.TimerRequest) -> None:
    """
    Nuevo timer trigger para procesar documentos de SharePoint con tablas
    """
    logging.info('*' * 80)
    logging.info('*** DOCUMENT PROCESSOR TIMER STARTED ***')
    logging.info(f'*** Execution time: {datetime.datetime.utcnow().isoformat()} ***')
    logging.info('*' * 80)
    
    try:
        # Configuration variable to control index clearing
        CLEAR_INDEX_ON_START = os.getenv("CLEAR_INDEX_ON_START", "false").lower() == "true"
        
        # Cargar configuración
        DOMINIO = os.getenv("SHAREPOINT_DOMAIN")
        HOSTNAME = os.getenv("SHAREPOINT_HOSTNAME")
        SITE = os.getenv("SHAREPOINT_SITE")
        FOLDER_PATH = os.getenv("SITE_FOLDER_PATH_SHAREPONT")
        TENANT_ID = os.getenv("AZURE_TENANT_ID")
        CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
        CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
        SCOPE = "https://graph.microsoft.com/.default"
        
        # 0. Clear index if requested
        if CLEAR_INDEX_ON_START:
            logging.info("=== STEP 0: CLEARING INDEX ===")
            try:
                # Search for all documents in the index
                search_results = search_client.search("*", select="id")
                document_ids = [doc["id"] for doc in search_results]
                
                if document_ids:
                    logging.info(f"Found {len(document_ids)} documents to delete from index")
                    
                    # Create delete actions for all documents
                    delete_actions = [{"@search.action": "delete", "id": doc_id} for doc_id in document_ids]
                    
                    # Batch delete in chunks of 1000 (Azure Search limit)
                    batch_size = 1000
                    for i in range(0, len(delete_actions), batch_size):
                        batch = delete_actions[i:i + batch_size]
                        search_client.upload_documents(documents=batch)
                        logging.info(f"Deleted batch of {len(batch)} documents")
                    
                    logging.info(f"Successfully cleared {len(document_ids)} documents from index")
                else:
                    logging.info("No documents found in index to delete")
                    
            except Exception as e:
                logging.error(f"Error clearing index: {e}")
        else:
            logging.info("Index clearing is disabled (CLEAR_INDEX_ON_START=false)")
        
        # 1. Leer last_files.json desde Azure Storage
        logging.info("=== STEP 1: READING LAST_FILES.JSON ===")
        last_files = []
        try:
            if blob_exists("last_files.json"):
                last_files = read_blob_json("last_files.json")
                logging.info(f"Found {len(last_files)} previously processed files")
            else:
                logging.info("No last_files.json found, starting fresh")
        except Exception as e:
            logging.error(f"Error reading last_files.json: {e}")
            last_files = []
        
        # 2. Acceder a SharePoint
        logging.info("=== STEP 2: ACCESSING SHAREPOINT ===")
        access_token = shp_get_access_token(TENANT_ID, CLIENT_ID, CLIENT_SECRET, SCOPE)
        site_info = shp_get_site_id(access_token, DOMINIO, SITE)
        site_id = site_info["id"]
        drive_id = shp_get_drive_id(access_token, site_id)
        
        logging.info(f"Connected to SharePoint site: {SITE}")
        logging.info(f"Site ID: {site_id}")
        logging.info(f"Drive ID: {drive_id}")
        
        # 3. Listar archivos y filtrar nuevos
        logging.info("=== STEP 3: LISTING AND FILTERING FILES ===")
        file_count, all_files = shp_list_drive_folder(access_token, drive_id, FOLDER_PATH)
        
        logging.info(f"Raw file listing from SharePoint (first 5 files):")
        for idx, file_path in enumerate(all_files[:5]):
            logging.info(f"  File {idx+1}: '{file_path}'")
        
        # Filtrar solo archivos PDF no procesados
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
        new_files = [f for f in pdf_files if f not in last_files]
        
        logging.info(f"Total files in SharePoint: {file_count}")
        logging.info(f"PDF files found: {len(pdf_files)}")
        logging.info(f"New PDF files to process: {len(new_files)}")
        logging.info(f"Processing fraction: {len(new_files)}/{len(pdf_files)} ({len(new_files)/max(len(pdf_files), 1)*100:.1f}%)")
        
        if len(new_files) > 0:
            logging.info(f"First new file to process: '{new_files[0]}'")
            logging.info(f"Example file path construction will be: 'https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{new_files[0]}:/content'")
        
        # Cargar prompt del vision agent
        vision_prompt = load_vision_agent_prompt()
        
        # 4. Procesar cada archivo nuevo
        processed_files = []
        for i, file_path in enumerate(new_files):
            logging.info('=' * 60)
            logging.info(f"*** PROCESSING FILE {i+1}/{len(new_files)}: {file_path} ***")
            logging.info('=' * 60)
            
            try:
                # Descargar archivo PDF
                logging.info(f"*** Attempting to download file: {file_path} ***")
                logging.info(f"Drive ID: {drive_id}")
                logging.info(f"Folder path context: {FOLDER_PATH}")
                
                # Use only alternative URL construction (more reliable)
                clean_file_path = file_path.strip('/')
                file_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{FOLDER_PATH}/{clean_file_path}:/content"
                logging.info(f"Download URL: {file_url}")
                
                headers = {"Authorization": f"Bearer {access_token}"}
                logging.info("Sending download request to Microsoft Graph API...")
                
                response = requests.get(file_url, headers=headers)
                
                logging.info(f"Download response status: {response.status_code}")
                logging.info(f"Response headers: {dict(response.headers)}")
                
                if response.status_code != 200:
                    logging.error(f"Failed to download file: {response.status_code}")
                    logging.error(f"Response content: {response.text[:500]}")  # First 500 chars of error
                    continue
                
                logging.info(f"Successfully downloaded file, size: {len(response.content)} bytes")
                
                # Guardar PDF temporal
                temp_pdf = tempfile.mktemp(suffix='.pdf')
                with open(temp_pdf, 'wb') as f:
                    f.write(response.content)
                
                # 4.1. Convertir primeras 3 páginas a imágenes usando EXACTAMENTE la lógica de opencv-tests.py
                logging.info("*** Converting PDF pages to images ***")
                
                # Abrir el PDF y procesar EXACTAMENTE como opencv-tests.py
                doc = fitz.open(temp_pdf)
                all_table_images = []
                
                for page_idx in range(min(3, len(doc))):
                    logging.info(f"*** Processing page {page_idx + 1} ***")
                    
                    page = doc.load_page(page_idx)
                    pix = page.get_pixmap(dpi=300)
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height,
                        pix.width,
                        pix.n
                    )
                    
                    if page_idx == 0:  # Special handling for the first page - EXACTAMENTE como opencv-tests.py
                        logging.info("*** Special handling for first page ***")
                        # Crop the first page into two parts
                        first_crop = img_data[0:1217, :]
                        second_crop = img_data[1218:, :]
                        
                        # Process the first cropped image
                        logging.info("*** Processing first page part 1 ***")
                        target_table = detect_first_page_table(first_crop)
                        if target_table:
                            x, y, w, h = target_table
                            cropped_table = first_crop[y:y+h, x:x+w]
                            temp_table_path = tempfile.mktemp(suffix=f'_{file_path}_page_1_part1_table.png')
                            cv2.imwrite(temp_table_path, cropped_table)
                            
                            # Save to blob storage
                            blob_name = f"temp_files/{file_path}_page_1_part1_table.png"
                            with open(temp_table_path, 'rb') as f:
                                write_blob_file(blob_name, f.read())
                            all_table_images.append((temp_table_path, blob_name))
                            logging.info(f"Found and saved table from first page part 1")
                        
                        # Process the second cropped image
                        logging.info("*** Processing first page part 2 ***")
                        tables_part2 = detect_and_crop_tables(second_crop)
                        logging.info(f"Found {len(tables_part2)} tables in first page part 2")
                        
                        for j, (x, y, w, h) in enumerate(tables_part2):
                            cropped_table = second_crop[y:y+h, x:x+w]
                            temp_table_path = tempfile.mktemp(suffix=f'_{file_path}_page_1_part2_table_{j}.png')
                            cv2.imwrite(temp_table_path, cropped_table)
                            
                            # Save to blob storage
                            blob_name = f"temp_files/{file_path}_page_1_part2_table_{j}.png"
                            with open(temp_table_path, 'rb') as f:
                                write_blob_file(blob_name, f.read())
                            all_table_images.append((temp_table_path, blob_name))
                    
                    else:  # Pages 2 and 3 - EXACTAMENTE como opencv-tests.py
                        logging.info(f"*** Processing page {page_idx + 1} (standard processing) ***")
                        # Convert RGBA to BGR if needed - EXACTAMENTE como opencv-tests.py
                        if pix.n == 4:
                            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                        
                        tables = detect_and_crop_tables(img_data)
                        logging.info(f"Found {len(tables)} tables in page {page_idx + 1}")
                        
                        for j, (x, y, w, h) in enumerate(tables):
                            cropped_table = img_data[y:y+h, x:x+w]
                            temp_table_path = tempfile.mktemp(suffix=f'_{file_path}_page_{page_idx + 1}_table_{j}.png')
                            cv2.imwrite(temp_table_path, cropped_table)
                            
                            # Save to blob storage
                            blob_name = f"temp_files/{file_path}_page_{page_idx + 1}_table_{j}.png"
                            with open(temp_table_path, 'rb') as f:
                                write_blob_file(blob_name, f.read())
                            all_table_images.append((temp_table_path, blob_name))
                
                doc.close()
                logging.info(f"*** Total tables detected for document: {len(all_table_images)} ***")
                
                # 4.6, 4.7, 4.8. Procesar cada imagen de tabla
                for idx, (table_img_path, blob_name) in enumerate(all_table_images):
                    logging.info(f"*** Processing table image {idx + 1}/{len(all_table_images)} ***")
                    
                    try:
                        # 4.6. Llamar al vision agent con reintentos automáticos
                        vision_response = call_vision_agent(table_img_path, vision_prompt, max_retries=3)
                        if not vision_response:
                            logging.error(f"Vision agent failed after all retry attempts for table {idx + 1}")
                            continue
                        
                        # Parsear respuesta JSON del vision agent (ya validada en call_vision_agent)
                        try:
                            # Clean the response to remove markdown formatting
                            cleaned_response = clean_json_response(vision_response)
                            vision_data = json.loads(cleaned_response)
                            logging.info(f"Successfully parsed vision agent JSON response for table {idx + 1}")
                        except json.JSONDecodeError as e:
                            # This should rarely happen now since we validate JSON in call_vision_agent
                            logging.error(f"Unexpected JSON error after retries: {e}")
                            logging.error(f"Cleaned response: {clean_json_response(vision_response)[:500]}...")
                            continue
                        
                        # 4.7.1. Procesar con Document Intelligence
                        doc_intel_content = process_with_document_intelligence(table_img_path)
                        
                        # 4.7.2. Generar embedding
                        if doc_intel_content:
                            # Usar deployment name que existe en Azure OpenAI
                            embeddings = generate_embeddings([doc_intel_content])
                            embedding = embeddings[0] if embeddings else []
                        else:
                            embedding = []
                        
                        # 4.8. Crear JSON object
                        # Extraer fecha del contenido
                        release_date = extract_date_from_text(doc_intel_content + " " + vision_data.get("markdown_content", ""))
                        
                        # Formato correcto para Azure Search DateTimeOffset (sin microsegundos)
                        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        document_url = f"https://{HOSTNAME}/sites/{SITE}/_layouts/15/Doc.aspx?sourcedoc={{doc_id}}&file={file_path}"
                        
                        search_document = {
                            "id": str(uuid.uuid4()),
                            "metadata_spo_item_name": os.path.basename(file_path),
                            "metadata_spo_item_path": document_url,
                            "metadata_spo_item_created_at": now,
                            "metadata_spo_item_last_modified": now,
                            "metadata_spo_item_release_date": release_date,
                            "metadata_spo_item_table_title": vision_data.get("metadata_spo_item_table_title", ""),
                            "markdown_content": vision_data.get("markdown_content", ""),
                            "content_description": vision_data.get("content_description", ""),
                            "embedding": embedding
                        }
                        
                        # 4.8. Indexar en Azure Search
                        search_client.upload_documents(documents=[search_document])
                        logging.info(f"*** INDEXED DOCUMENT IN AZURE SEARCH ***")
                        logging.info(f"Document ID: {search_document['id']}")
                        logging.info(f"Table title: {search_document['metadata_spo_item_table_title']}")
                        
                    except Exception as e:
                        logging.error(f"Error processing table image {idx + 1}: {e}")
                        continue
                
                # Agregar archivo a la lista de procesados
                processed_files.append(file_path)
                logging.info(f"*** COMPLETED PROCESSING FILE: {file_path} ***")
                
                # Limpiar archivos temporales
                try:
                    os.unlink(temp_pdf)
                    for table_img_path, _ in all_table_images:
                        if os.path.exists(table_img_path):
                            os.unlink(table_img_path)
                except Exception as e:
                    logging.warning(f"Error cleaning temporary files: {e}")
                
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue
        
        # 5. Actualizar last_files.json
        if processed_files:
            logging.info("=== STEP 5: UPDATING LAST_FILES.JSON ===")
            updated_last_files = last_files + processed_files
            write_blob_json("last_files.json", updated_last_files)
            logging.info(f"Updated last_files.json with {len(processed_files)} new files")
        
        logging.info('*' * 80)
        logging.info(f'*** DOCUMENT PROCESSOR TIMER COMPLETED ***')
        logging.info(f'*** Processed {len(processed_files)} files successfully ***')
        logging.info('*' * 80)
        
    except Exception as e:
        logging.error(f"Critical error in document processor timer: {e}")
        logging.info('*' * 80)
        logging.info('*** DOCUMENT PROCESSOR TIMER FAILED ***')
        logging.info('*' * 80)

# Reduce el nivel de logging de Azure y requests para evitar ruido en consola
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.storage').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)