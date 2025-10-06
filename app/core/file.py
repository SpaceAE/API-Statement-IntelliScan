import io
from typing import BinaryIO, Optional

from pypdf import PdfReader


class IncorrectPasswordException(Exception):
	pass


class PasswordRequiredException(Exception):
	pass


def read_file(file: BinaryIO, password: Optional[str]) -> bytes:
	"""
	อ่านไฟล์อัปโหลดเป็น bytes
	- ถ้าเป็น PDF ที่เข้ารหัส: ตรวจรหัสผ่าน (ถ้าไม่ให้/ให้ผิด -> raise)
	- คืน bytes เพื่อให้สาขา PDF/CSV/XLSX ใช้ต่อได้
	"""
	data = file.read()

	# ลองเปิดด้วย PdfReader เพื่อตรวจว่าเป็น PDF และ encrypted ไหม
	try:
		reader = PdfReader(io.BytesIO(data))
		if reader.is_encrypted:
			if not password:
				raise PasswordRequiredException(
					'Password is required for this encrypted PDF.'
				)
			ok = reader.decrypt(password)
			if not ok:
				raise IncorrectPasswordException(
					'Incorrect password for the encrypted PDF.'
				)
	except Exception:
		# ไม่ใช่ PDF ก็ไม่เป็นไร ปล่อยให้สาขา CSV/XLSX ไปจัดการต่อ
		pass

	return data
