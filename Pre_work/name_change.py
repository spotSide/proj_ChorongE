import os
from tkinter import Tk, filedialog

def rename_jpg_files(directory, new_name):
    try:
        # 디렉토리 내부 파일 목록 가져오기
        files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
        
        if not files:
            print("해당 디렉토리에 JPG 파일이 없습니다.")
            return
        
        # 파일 이름을 바꾸기
        for index, file in enumerate(files, start=1):
            old_path = os.path.join(directory, file)
            new_filename = f"{new_name}_{index}.jpg"
            new_path = os.path.join(directory, new_filename)
            
            os.rename(old_path, new_path)
            print(f"{file} -> {new_filename} 로 이름 변경 완료")
        
        print("모든 파일의 이름 변경이 완료되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

# 디렉토리 경로 선택
Tk().withdraw()  # GUI 창 숨기기
print("디렉토리를 선택하세요.")
directory = filedialog.askdirectory()

if directory:
    new_name = input("새로운 이름 형식을 입력하세요: ")
    rename_jpg_files(directory, new_name)
else:
    print("디렉토리가 선택되지 않았습니다.")
