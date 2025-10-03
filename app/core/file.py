from typing import Optional

from pypdf import PdfReader


class IncorrectPasswordException(Exception):
	pass


class PasswordRequiredException(Exception):
	pass


def read_file(file, password: Optional[str]):
	reader = PdfReader(file)
	if reader.is_encrypted:
		if not password:
			raise PasswordRequiredException(
				'Password is required for this encrypted PDF.'
			)
		if not reader.decrypt(password):
			raise IncorrectPasswordException(
				'Incorrect password for the encrypted PDF.'
			)
	return reader.get_page(0).extract_text()
