## import pandas as pd
#import os
#import win32com.client
#from datetime import datetime, timedelta
#outlook = win32com.client.Dispatch('outlook.application')
#mapi = outlook.GetNamespace("MAPI")
#for account in mapi.Accounts:
#	print(account.DeliveryStore.DisplayName)
#inbox = mapi.GetDefaultFolder(6)
#messages = inbox.Items"


## for sending the mails through python
from itertools import groupby
import win32com.client as win32
import os
# library for using applications

olApp = win32.Dispatch('Outlook.Application')
#MAPI- messaging application programming interface
olNS = olApp.GetNameSpace('MAPI')
#mail creation
mailItem= olApp.CreateItem(0)
mailItem.Subject= 'outlook email send using python' #subject inclusion
mailItem.BodyFormat = 1 
mailItem.Body = 'test-email'  #mail content
mailItem.To = 'krishanu.kundu@infra.market'  #receipent
#choose account to send the email
mailItem._oleobj_.Invoke(*(64209,0 , 8, 0, olNS.Accounts.Item('bilal.khan@infra.market'))) 
mailItem.Attachments.Add(os.path.join(os.getcwd(), 'Catalogue_0034.xlsx'))
mailItem.Display()
mailItem.Save()
mailItem.Send()

