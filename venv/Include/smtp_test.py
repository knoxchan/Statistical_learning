import smtplib
from email.mime.text import MIMEText
from email.header import Header

sender = 'knox.chan@asiapac.com.cn'
receivers = ['173687872@qq.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

# 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
message = MIMEText('Python 邮件发送测试...', 'plain', 'utf-8')
message['From'] = Header("菜鸟教程", 'utf-8')  # 发送者
message['To'] = Header("测试", 'utf-8')  # 接收者

subject = 'Python SMTP 邮件测试'
message['Subject'] = Header(subject, 'utf-8')

try:
    smtpObj = smtplib.SMTP()
    smtpObj.connect('mail.asiapac.com.cn','25')
    # smtpObj.starttls()
    smtpObj.login('knox.chan@asiapac.com.cn','T3~H5y#6f7U')
    smtpObj.sendmail(sender, receivers, message.as_string())
    print("邮件发送成功")
except Exception as e:
    print(e)
# except smtplib.SMTPException:
#     print("Error: 无法发送邮件")
