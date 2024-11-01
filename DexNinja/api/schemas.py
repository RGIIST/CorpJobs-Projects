from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class FileMetaData(BaseModel):
    id: str = Field(default="")
    name: str = Field(default="")
    contentType: str = Field(default="")
    url: str = Field(default="")
    # size: str
    # data: List[bytes]
    containerName: str = Field(default="")

class Document(BaseModel):
    id: str = Field(default="")
    organisationId: str = Field(default="")
    workqueueId: str = Field(default="")
    name: str = Field(default="")
    groupId: str = Field(default="")
    fileData: FileMetaData
    documentImages: List[FileMetaData] = Field(default=list())
    state: str = Field(default="")
    text: str = Field(default="")
    contentType: str = Field(default="") # TIFF,PDF,JPG etc
    documentType: str = Field(default="") # Classification value - contract,invoice etc.
    createdOn: str = Field(default="")
    updatedOn: str = Field(default="")
    hasException: bool = Field(default=False)
    entities: List[str] = Field(default=list())

    
class DocumentText(BaseModel):
    text: str


class countryName(str, Enum):
    India = "india"
    Pakistan = "pak"
    Nepal = "nepal"
