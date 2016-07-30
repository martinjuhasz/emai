from emai.persistence import Recording, Bag, Message, load_json, to_objectid, get_async_file_descriptor
import asyncio
from bson import ObjectId

async def main():
    bags = Bag.find({'label': {'$gte': 2}})
    for bag in (await bags.to_list(None)):
        for message_id in bag.messages:
            message = await Message.find_one({'_id': ObjectId(message_id)})
            if not message.label:
                message.label = bag.label
                await message.commit()
            elif message.label != bag.label:
                print('corrupt', message.label, bag.label, message.id)
            else:
                pass
                # print('duplicate')





if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    loop.run_until_complete(main())
