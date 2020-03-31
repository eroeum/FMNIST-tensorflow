import os
import time
from twilio.rest import Client
from datetime import datetime

class Notifier(object):
    def __init__(self, number):
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        from_number = os.environ.get('TWILIO_SMS_NUMBER')

        if(account_sid == "" or auth_token == "" or from_number == ""):
            msg = "Error: Environment variables unset.  Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_SMS_NUMBER"
            raise Exception(msg)


        self.client = Client(account_sid, auth_token)

        self.from_number = from_number
        self.number = number
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = datetime.now()

    def end(self):
        self.end_time = datetime.now()

    def duration(self):
        if(self.start_time == None or self.end_time == None):
            return -1
        else:
            return self.end_time - self.start_time

    def notify_acc(self, acc):
        msg = "Training Complete:\n" + \
              "Duration:" + str(self.duration()) + "\n" + \
              "Accuracy: " + str(acc)
        self.client.messages.create(to=self.number,
                               from_=self.from_number,
                               body=msg)

if __name__ == '__main__':
    test = Notifier("+13013510464")
    test.start()
    time.sleep(5)
    test.end()
    test.notify_acc(63.27)
