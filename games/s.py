import shutil, tempfile
shutil.rmtree(tempfile.gettempdir() + "/torch_extensions", ignore_errors=True)
print("✅ torch_extensions 폴더 삭제 완료!")
