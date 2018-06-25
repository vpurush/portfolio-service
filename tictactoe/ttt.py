import tensorflow as tf
import numpy as np
import math
import boto3
import os
import json
import zipfile

input = [
    [
        [-1., 0., -1.],
        [0., 1., 0.],
        [0., 0., 0.]
    ],
    [
        [1., -1., 1.],
        [0., -1., 0.],
        [0., 0., 0.]
    ],
    [
        [1., -1., 1.],
        [1., -1., -1.],
        [0., 0., 0.]
    ],
    [
        [-1., 1., -1.],
        [0., 1., 0.],
        [0., -1., 0.]
    ],
    [
        [1., 0., 1.],
        [0., -1., -1.],
        [0., 0., 0.]
    ]
];

output = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0]
];

def train(event, context):
    graph = tf.Graph()

    with graph.as_default():
        x = tf.placeholder("float32", shape=[None, 3, 3], name="x")
        y = tf.placeholder("float32", shape=[None, 9], name="y")

        x_reshaped = tf.reshape(x, shape=[-1, 9])
        neurons_output = 9



        with tf.Session(graph=graph) as sess:

            y_ = tf.layers.dense(x_reshaped, neurons_output, activation=tf.nn.tanh, kernel_initializer=tf.initializers.constant(.1, tf.float32), bias_initializer=tf.initializers.constant(0.1, tf.float32), name="output")

            loss = tf.reduce_mean(tf.square(tf.add(y_, tf.negative(y))))

            optimizer = tf.train.GradientDescentOptimizer(0.1)
            # optimizer = tf.train.AdamOptimizer()

            train = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            sess.run(init)

            for j in range(100):
                # sess.run(train, feed_dict={x: [input[0]], y: [output[0]] })
                sess.run(train, feed_dict={x: input, y: output })
            

            saver.save(sess, "/tmp/ttt/nn_model")

            print("y", sess.run(y, feed_dict={y:[output[0]] }))

            print("y_", sess.run(y_, feed_dict={x: [input[0]] }))

            print("argmax", np.argmax(sess.run(y_, feed_dict={x: [input[0]] })))

            print("add", sess.run(tf.add(y_, tf.negative(y)), feed_dict={x: [input[0]], y:[output[0]] }))

            print("square", sess.run(tf.square(tf.add(y_, tf.negative(y))), feed_dict={x: [input[0]], y:[output[0]] }))

            print("loss first", sess.run(loss, feed_dict={x: [input[0]], y:[output[0]] }))

            print("loss all", sess.run(loss, feed_dict={x: input, y:output }))

            

            print(np.argmax(sess.run(y_, feed_dict={x: [input[0]] })), output[0])

            print(np.argmax(sess.run(y_, feed_dict={x: [input[1]] })), output[1])

            print(np.argmax(sess.run(y_, feed_dict={x: [input[2]] })), output[2])

            print(np.argmax(sess.run(y_, feed_dict={x: [input[3]] })), output[3])

            print(np.argmax(sess.run(y_, feed_dict={x: [input[4]] })), output[4])

        

            sess.close()

    print("Dir ",os.listdir("/tmp/ttt"))
    # with open('/tmp/ttt/checkpoint', 'r') as content_file:
    #     content = content_file.read()
    #     print("content", content)

    saveFiles("/tmp/ttt")
    return {
        "statusCode": 200,
        "body": json.dumps({"success": True, "result": "Done"}),
        "headers":{
            "Access-Control-Allow-Origin": os.environ["ALLOW_ORIGIN"]
        }
    }

def saveFiles(path):

    (zfName, zfNameWithoutPath) = createArchive(path)
    print("zfName", zfName)

    #s3 = boto3.client('s3')
    #bucket_name = 'vpurush-ttt'

    #s3.upload_file(zfName, bucket_name, zfNameWithoutPath)


def createArchive(path):
    zfNameWithoutPath = 'ttt.gz'
    zfName = '/tmp/' + zfNameWithoutPath
    zfile = zipfile.ZipFile(zfName, 'w')

    # Adding files from directory 'files'
    for root, dirs, files in os.walk(path):
        for f in files:
            zfile.write(os.path.join(root, f))

    zfile.close()
    return (zfName, zfNameWithoutPath)



def cleanFolder(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.remove(os.path.join(root, f))

    os.rmdir(path)

def downloadAndExtract():
    if not os.path.exists("/tmp/ttt"):
        print("path does not exist downloading")
        s3 = boto3.resource('s3')
        s3.Bucket('vpurush-ttt').download_file('ttt.gz', '/tmp/ttt.gz')

        zfile = zipfile.ZipFile('/tmp/ttt.gz', "r")
        zfile.extractall('/')
        zfile.close()
    else:
        print("path exist. Skipping download")
    
    print("Dir ",os.listdir("/tmp/ttt"))

def restoreModel(board):
    downloadAndExtract()

    x = tf.placeholder("float32", shape=[None, 3, 3], name="x")
    y = tf.placeholder("float32", shape=[None, 9], name="y")

    x_reshaped = tf.reshape(x, shape=[-1, 9])
    neurons_output = 9


    result = None

    with tf.Session() as sess:

        y_ = tf.layers.dense(x_reshaped, neurons_output, activation=tf.nn.tanh, kernel_initializer=tf.initializers.constant(.1, tf.float32), bias_initializer=tf.initializers.constant(0.1, tf.float32), name="output")

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("/tmp/ttt"))

        print("Model restored")

        result = np.argmax(sess.run(y_, feed_dict={x: [board] }))
        print(result, output[0])

    return result



def nextMove(event, context):
    if(event['queryStringParameters'] == None or event['queryStringParameters']['board'] == None):
        return {
            "statusCode": 200,
            "body": json.dumps({"success": False, "error": "No Input"}),
            "headers":{
                "Access-Control-Allow-Origin": os.environ["ALLOW_ORIGIN"]
            }
        }
    else:
        try:
            input = event['queryStringParameters']['board'][0]
            board = eval(input)
            result = restoreModel(board)
            print("board", board)
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"success": False, "error": "Error occurred"}),
                "headers":{
                    "Access-Control-Allow-Origin": os.environ["ALLOW_ORIGIN"]
                }
            }
        else:            
            print("result is ", result)
            return {
                "statusCode": 200,
                "body":  json.dumps({"success": True, "result": str(result)}),
                "headers":{
                    "Access-Control-Allow-Origin": os.environ["ALLOW_ORIGIN"]
                }
            }
