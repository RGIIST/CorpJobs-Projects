from fastapi import APIRouter
from api.schemas import Document, countryName
from prediction import process
# from logger_conf.logger import get_logger, CYAN, PURPLE, END

# logger = get_logger(__name__)
router = APIRouter()

# -------------------------------------------------------------------------------------------------------
@router.get("/ping", response_description="pinging")
async def ping_me():
    """Ping
    """
    return {"Status": "Healthy"}

# -------------------------------------------------------------------------------------------------------
@router.post("/extract/passport", response_description="Extraction of Passport documents")
async def labels_with_text(country: countryName, doc: Document):
    """
    """

    entities = process(doc.fileData.url, country.value) #ndarray
    return entities

# +++++++++++++++++++++++++++++++++++++++++++++++++ END +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++