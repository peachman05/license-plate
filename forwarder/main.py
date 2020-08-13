import zmq

def main():

    try:
        context = zmq.Context(1)
        # Socket facing clients
        frontend = context.socket(zmq.PULL)
        frontend.bind("tcp://*:5563")
        
        # Socket facing services
        backend = context.socket(zmq.PUSH)
        backend.bind("tcp://*:5564")

        zmq.device(zmq.STREAMER, frontend, backend)
    except Exception as err:
        print(err)
        print("bringing down zmq device")
    finally:
        pass
        frontend.close()
        backend.close()
        context.term()

if __name__ == "__main__":
    main()