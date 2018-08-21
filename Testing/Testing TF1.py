import os
import tensorflow as tf

intNum1 = float(input("enter num 1: "))
intNum2 = float(input("enter num 2: "))

num1 = tf.Variable(intNum1, name="num1")
num2 = tf.Variable(intNum2, name="num2")

sum = tf.add(num1, num2, name="sum")
print("tf sum: " + str(sum))

globalVI = tf.global_variables_initializer()

with tf.Session() as session:
    globalVI.run()
    result = sum.eval()

print("result : " + str(result))