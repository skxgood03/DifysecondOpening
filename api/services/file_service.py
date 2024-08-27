import datetime
import hashlib
import uuid
from collections.abc import Generator
from typing import Union

from flask_login import current_user
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import NotFound

from configs import dify_config
from core.file.upload_file_parser import UploadFileParser
from core.rag.extractor.extract_processor import ExtractProcessor
from extensions.ext_database import db
from extensions.ext_storage import storage
from models.account import Account
from models.model import EndUser, UploadFile
from services.errors.file import FileTooLargeError, UnsupportedFileTypeError

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "gif", "svg"]
IMAGE_EXTENSIONS.extend([ext.upper() for ext in IMAGE_EXTENSIONS])

ALLOWED_EXTENSIONS = ["txt", "markdown", "md", "pdf", "html", "htm", "xlsx", "xls", "docx", "csv"]
UNSTRUCTURED_ALLOWED_EXTENSIONS = [
    "txt",
    "markdown",
    "md",
    "pdf",
    "html",
    "htm",
    "xlsx",
    "xls",
    "docx",
    "csv",
    "eml",
    "msg",
    "pptx",
    "ppt",
    "xml",
    "epub",
]

PREVIEW_WORDS_LIMIT = 3000


class FileService:
    @staticmethod
    def upload_file(file: FileStorage, user: Union[Account, EndUser], only_image: bool = False) -> UploadFile:
        # 获取文件名和扩展名
        filename = file.filename
        extension = file.filename.split('.')[-1]
        # 如果文件名过长，截断并保留扩展名
        if len(filename) > 200:
            filename = filename.split('.')[0][:200] + '.' + extension
        # 根据配置获取允许的文件类型
        etl_type = dify_config.ETL_TYPE
        allowed_extensions = (
            UNSTRUCTURED_ALLOWED_EXTENSIONS + IMAGE_EXTENSIONS
            if etl_type == "Unstructured"
            else ALLOWED_EXTENSIONS + IMAGE_EXTENSIONS
        )
        # 检查文件类型是否在允许的范围内
        if extension.lower() not in allowed_extensions:
            raise UnsupportedFileTypeError()
        elif only_image and extension.lower() not in IMAGE_EXTENSIONS:
            raise UnsupportedFileTypeError()

        #  # 读取文件内容
        file_content = file.read()

        # 获取文件大小
        file_size = len(file_content)
        # 设置文件大小限制，图片文件和非图片文件有不同的限制
        if extension.lower() in IMAGE_EXTENSIONS:
            file_size_limit = dify_config.UPLOAD_IMAGE_FILE_SIZE_LIMIT * 1024 * 1024
        else:
            file_size_limit = dify_config.UPLOAD_FILE_SIZE_LIMIT * 1024 * 1024
        # 检查文件大小是否超过限制
        if file_size > file_size_limit:
            message = f"File size exceeded. {file_size} > {file_size_limit}"
            raise FileTooLargeError(message)

        # 生成文件的唯一UUID
        file_uuid = str(uuid.uuid4())
        # 确定当前租户ID
        if isinstance(user, Account):
            current_tenant_id = user.current_tenant_id
        else:
            # end_user
            current_tenant_id = user.tenant_id
        # 构建文件在存储系统中的路径
        file_key = "upload_files/" + current_tenant_id + "/" + file_uuid + "." + extension

        #  # 保存文件到存储系统
        storage.save(file_key, file_content)

        #  # 创建UploadFile模型实例，并填充必要的字段
        upload_file = UploadFile(
            tenant_id=current_tenant_id,
            storage_type=dify_config.STORAGE_TYPE,
            key=file_key,
            name=filename,
            size=file_size,
            extension=extension,
            mime_type=file.mimetype,
            created_by_role=("account" if isinstance(user, Account) else "end_user"),
            created_by=user.id,
            created_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            used=False,
            hash=hashlib.sha3_256(file_content).hexdigest(),
        )

        db.session.add(upload_file)
        db.session.commit()

        return upload_file

    @staticmethod
    def upload_text(text: str, text_name: str) -> UploadFile:
        if len(text_name) > 200:
            text_name = text_name[:200]
        # user uuid as file name
        file_uuid = str(uuid.uuid4())
        file_key = "upload_files/" + current_user.current_tenant_id + "/" + file_uuid + ".txt"

        # save file to storage
        storage.save(file_key, text.encode("utf-8"))

        # save file to db
        upload_file = UploadFile(
            tenant_id=current_user.current_tenant_id,
            storage_type=dify_config.STORAGE_TYPE,
            key=file_key,
            name=text_name,
            size=len(text),
            extension="txt",
            mime_type="text/plain",
            created_by=current_user.id,
            created_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            used=True,
            used_by=current_user.id,
            used_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
        )

        db.session.add(upload_file)
        db.session.commit()

        return upload_file

    @staticmethod
    def get_file_preview(file_id: str) -> str:
        upload_file = db.session.query(UploadFile).filter(UploadFile.id == file_id).first()

        if not upload_file:
            raise NotFound("File not found")

        # extract text from file
        extension = upload_file.extension
        etl_type = dify_config.ETL_TYPE
        allowed_extensions = UNSTRUCTURED_ALLOWED_EXTENSIONS if etl_type == "Unstructured" else ALLOWED_EXTENSIONS
        if extension.lower() not in allowed_extensions:
            raise UnsupportedFileTypeError()

        text = ExtractProcessor.load_from_upload_file(upload_file, return_text=True)
        text = text[0:PREVIEW_WORDS_LIMIT] if text else ""

        return text

    @staticmethod
    def get_image_preview(file_id: str, timestamp: str, nonce: str, sign: str) -> tuple[Generator, str]:
        result = UploadFileParser.verify_image_file_signature(file_id, timestamp, nonce, sign)
        if not result:
            raise NotFound("File not found or signature is invalid")

        upload_file = db.session.query(UploadFile).filter(UploadFile.id == file_id).first()

        if not upload_file:
            raise NotFound("File not found or signature is invalid")

        # extract text from file
        extension = upload_file.extension
        if extension.lower() not in IMAGE_EXTENSIONS:
            raise UnsupportedFileTypeError()

        generator = storage.load(upload_file.key, stream=True)

        return generator, upload_file.mime_type

    @staticmethod
    def get_public_image_preview(file_id: str) -> tuple[Generator, str]:
        upload_file = db.session.query(UploadFile).filter(UploadFile.id == file_id).first()

        if not upload_file:
            raise NotFound("File not found or signature is invalid")

        # extract text from file
        extension = upload_file.extension
        if extension.lower() not in IMAGE_EXTENSIONS:
            raise UnsupportedFileTypeError()

        generator = storage.load(upload_file.key)

        return generator, upload_file.mime_type
