import io
from typing import BinaryIO, Optional

from pypdf import PdfReader


class IncorrectPasswordException(Exception):
	pass


class PasswordRequiredException(Exception):
	pass


def read_file(file: BinaryIO, password: Optional[str]) -> bytes:
	data = file.read()

	# ตรวจด้วย magic header ก่อนว่าเป็น PDF จริง
	is_probably_pdf = data[:5] == b'%PDF-'

	if is_probably_pdf:
		try:
			reader = PdfReader(io.BytesIO(data))
			if reader.is_encrypted:
				if not password:
					raise PasswordRequiredException('Password is required for PDF.')
				if not reader.decrypt(password):
					raise IncorrectPasswordException('Incorrect password for PDF.')
		except IncorrectPasswordException:
			raise
		except PasswordRequiredException:
			raise
		except Exception:
			# ถ้าอ่านโครงสร้าง pdf ไม่ได้ ปล่อยให้ไปแตกแขนงต่อ (จะไปตก PARSE_FAILED ทีหลัง)
			pass

	return data
