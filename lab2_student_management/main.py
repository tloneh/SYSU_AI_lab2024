from studata import *

file_name = "student_data.txt"
out_file = "export_data.txt"

students = StuData(file_name)

students.AddData("hugy", '001', 'M', '20')

students.SortData('name')
# students.SortData('stu_num')
# students.SortData('gender')
# students.SortData('age')
# print(students.data)

students.ExportFile(out_file)
