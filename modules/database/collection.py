from pymongo import MongoClient


client = MongoClient("mongodb+srv://admin:admin1234@together.cvq6ffb.mongodb.net/?retryWrites=true&w=majority")

db = client.seedlings
seedlings = db.seedlings
pollution_report = db.report
crops = db.crop
flowers = db.flower
leaves = db.leaf
paddy = db.paddy
weeds = db.weed


def addSeedlingImage(i_name, Type, time, url):
    seedlings.insert_one({
        "file_name": i_name,
        "prediction": Type,
        "upload_time": time,
        "url": url
    })


def addFlowerImage(i_name, Type, time, url):
    flowers.insert_one({
        "file_name": i_name,
        "prediction": Type,
        "upload_time": time,
        "url": url
    })


def addLeaveImage(i_name, Type, time, url):
    leaves.insert_one({
        "file_name": i_name,
        "prediction": Type,
        "upload_time": time,
        "url": url
    })


def addWeedImage(i_name, Type, time, url):
    weeds.insert_one({
        "file_name": i_name,
        "prediction": Type,
        "upload_time": time,
        "url": url
    })


def addPaddyImage(i_name, Type, time, url):
    paddy.insert_one({
        "file_name": i_name,
        "prediction": Type,
        "upload_time": time,
        "url": url
    })


def cropRecommendation(Type, time):
    crops.insert_one({
        "prediction": Type,
        "upload_time": time
    })


def pollutionReport(image, time, location):
    pollution_report.insert_one({
        "prediction": image,
        "upload_time": time,
        "location": location
    })
