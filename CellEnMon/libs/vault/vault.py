import os
import sys
from ansible_vault import Vault

class VaultService:
    def __init__(self):
        if "ANSIBLE_MASTER_KEY" in os.environ:
            self.master_key=os.environ["ANSIBLE_MASTER_KEY"]

            vault=Vault(self.master_key)
            self.dict_secrets = vault.load(open('secrets.yaml').read())

        else:
            raise "ENVIRONMENT VARIABLE: ANSIBLE_MASTER_KEY is not set. Please set it to pull secrets"


if __name__=="__main__":
    vault_service = VaultService()
    if not sys.argv[1]:
        iter_dict=vault_service.dict_secrets[sys.argv[1]]
        try:
            for key in sys.argv[2:]:
                iter_dict=iter_dict[key]
        except ValueError:
            raise ValueError("Please provide a space seperated keys to the secrets you want to pull")

    print(iter_dict)


