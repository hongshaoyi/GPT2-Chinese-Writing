# GPT2-Chinese-Writing
还没时间细写，先写个简单流程(没时间排版，大家先将就看，有时间写个完整的)

原项目地址：https://github.com/Morizeyao/GPT2-Chinese
感谢原作者的劳动，我是在这个基础上学习改进的

如何使用：
在data目录下建立自己的语料目录，名字随意，然后子目录下放入你想要训练的文件，只要是python打得开的文本即可，比如txt，前提是utf8格式，可以放多个，比如你可以放全套的金庸小说进去
然后运行linux目录下的
./build_pretrain_file.sh 子目录名
它就会在pre_data目录下生成一个同子目录名的文件夹，里面就是vocab和config

接下来开始训练：
运行：./train.sh 目录名(pre_data下的目录名,无需加pre_data前缀)
可以改下train.sh的参数，比如训练轮数等
还有如果内存爆了，可以改下config文件，降低下n_ctx等

训练完后会在trained_model生成对应轮数的训练文件，最后一轮是final_model
如果想继续训练就把train.sh里的注释打开，就可以接着前面的训练继续训练下去

之后是生成:
运行: ./generate.sh
就会开始生成，可以改里面的参数来调整生成长度和数量等
默认是用快速生成，速度非常快，原作者的快速有bug，我修好了，但是有时候会生成很多回车的样本，不知道是不是bug，我还没有头绪，欢迎大家指出问题所在，这时候可以考虑用慢速模式，把注释去掉就行，会很慢，大家看着选择

最后是save_model.sh，这个是将final_model移动到model文件夹下
运行: ./save_model.sh 指定文件夹

这个项目我是指用来学习研究的，最开始是因为原作者不更新了，加上语料必须是json格式，这对小说来说简直是灾难，加上代码上有很多优化的空间，所以就进行了学习修改
实际上我本身也不会python，现学的，毕竟是程序员，学起来还是很快的
欢迎大家拿去玩，互相交流

ps:有空会提交详细的readme，大家先讲究看吧
