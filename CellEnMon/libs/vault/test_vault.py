from CellEnMon.libs.vault.vault import VaultService
import unittest

class TestVaultService(unittest.TestCase):
    def setUp(self) -> None:
        self.vault_service=VaultService()

    def test_extract_secret_correctly(self):
        self.assertEqual("xxx",self.vault_service.dict_secrets["test"])
