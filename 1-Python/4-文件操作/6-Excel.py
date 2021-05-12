import xlrd
import xlwt
from datetime import date,datetime

def set_style(name, height, bold=False):
    style = xlwt.XFStyle()  # 初始化样式

    font = xlwt.Font()  # 为样式创建字体
    font.name = name  # 'Times New Roman'
    font.bold = bold
    font.color_index = 4
    font.height = height

    # borders= xlwt.Borders()
    # borders.left= 6
    # borders.right= 6
    # borders.top= 6
    # borders.bottom= 6

    style.font = font
    # style.borders = borders

    return style

def read_excel():
    workbook = xlrd.open_workbook(r'F:\demo.xlsx')  # 打开文件

    print(workbook.sheet_names())   # 获取所有sheet [u'sheet1', u'sheet2']

    # 根据sheet索引或者名称获取sheet内容
    sheet2 = workbook.sheet_by_index(1)  # sheet索引从0开始
    sheet2 = workbook.sheet_by_name('sheet2')

    # sheet的名称，行数，列数
    print(sheet2.name, sheet2.nrows, sheet2.ncols)

    # 获取整行和整列的值（数组）
    rows = sheet2.row_values(3)  # 获取第四行内容
    cols = sheet2.col_values(2)  # 获取第三列内容
    print(rows)
    print(cols)

    # 获取单元格内容
    print(sheet2.cell(1, 0).value.encode('utf-8'))
    print(sheet2.cell_value(1, 0).encode('utf-8'))
    print(sheet2.row(1)[0].value.encode('utf-8'))

    # 获取单元格内容的数据类型
    print(sheet2.cell(1, 0).ctype)

# 写excel
def write_excel():
    f = xlwt.Workbook()  # 创建工作簿

    '''
    创建第一个sheet:sheet1
    '''
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    row0 = [u'业务', u'状态', u'北京', u'上海', u'广州', u'深圳', u'状态小计', u'合计']
    column0 = [u'机票', u'船票', u'火车票', u'汽车票', u'其它']
    status = [u'预订', u'出票', u'退票', u'业务小计']

    # 生成第一行
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    # 生成第一列(合并4行)
    # i, j = 1, 0
    # while i < 4 * len(column0) and j < len(column0):
    #     sheet1.write_merge(i, i + 3, 0, 0, column0[j], set_style('Arial', 220, True))  # 第一列
    #     sheet1.write_merge(i, i + 3, 7, 7)  # 最后一列"合计"
    #     i += 4
    #     j += 1

    sheet1.write_merge(21, 21, 0, 1, u'合计', set_style('Times New Roman', 220, True))

    # 生成第二列
    i = 0
    while i < 4 * len(column0):
        for j in range(0, len(status)):
            sheet1.write(j + i + 1, 1, status[j])
        i += 4

    f.save('demo1.xlsx')  # 保存文件


if __name__ == '__main__':
    # read_excel()
    write_excel()