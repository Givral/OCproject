import pandas
from openpyxl import load_workbook

# write header
h_ratio_w_h ='ratio_w_h'
h_area = 'area'
h_area_rect = 'area_rect'
h_ratio_area = 'ratio_area'
h_angle_of_rect = 'angle_of_rect'
h_angle_o_line = 'angle_o_line'
h_camera_region = 'camera_region'
h_frame_no = 'frame_no'
def write_header(book):
    #writing file header
    header = pandas.DataFrame({'ratio_w_h':[h_ratio_w_h], 'area':[h_area], 'area_rect':[h_area_rect], 'ratio_area':[h_ratio_area],
                'angle_of_rect':[h_angle_of_rect], 'angle_o_line':[h_angle_o_line],'camera_region':[h_camera_region],'frame_no':[h_frame_no]})
    writer = pandas.ExcelWriter('datatop0.xlsx', engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    header.to_excel(writer, startrow=1, header=False, index=False)
    writer.save()