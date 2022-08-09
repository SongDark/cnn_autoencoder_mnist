# coding:utf-8
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf 
from utils import *
from networks import *
from datamanager import datamanager_mnist

tf.compat.v1.disable_eager_execution()

class AutoEncoder_sup(BasicTrainFramework):
    def __init__(self, batch_size, version="AE_SUP"):
        super(AutoEncoder_sup, self).__init__(batch_size, version)
        
        self.data = datamanager_mnist(train_ratio=0.8, fold_k=None, expand_dim=True, norm=True)
        self.sample_data = self.data(self.batch_size, phase='test', var_list=["data", "labels"])

        self.noise_dim = 10
        self.class_num = 10
        self.emb_dim = self.noise_dim + self.class_num

        self.encoder = CNN_Encoder(output_dim=self.emb_dim, sn=False)
        self.decoder = CNN_Decoder(sn=False)

        self.build_placeholder()
        self.build_network()
        self.build_optimizer()

        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        self.source = tf.compat.v1.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        self.target = tf.compat.v1.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        self.labels = tf.compat.v1.placeholder(shape=(self.batch_size, self.class_num), dtype=tf.float32)
    
    def build_network(self):
        self.embedding = self.encoder(self.source, is_training=True, reuse=False)
        self.embedding_test = self.encoder(self.source, is_training=False, reuse=True)
        self.pred = self.decoder(self.embedding, is_training=True, reuse=False)
        self.pred_test = self.decoder(self.embedding, is_training=False, reuse=True)

        self.mean_pred, self.std_pred = tf.nn.moments(x=self.pred, axes=range(len(self.source.shape)))
        self.std_pred = tf.sqrt(self.std_pred)

    def build_optimizer(self):
        self.reconstruct_loss = mse(self.pred, self.target, self.batch_size)
        
        dist_code = self.embedding[:, :self.class_num]
        self.dist_code_test = tf.nn.softmax(self.embedding_test[:, :self.class_num])
        self.Q_loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=dist_code, labels=self.labels))
        
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.reconstruct_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.reconstruct_loss, var_list=self.encoder.vars + self.decoder.vars)
            self.Q_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.Q_loss, var_list=self.encoder.vars)

    def hist(self, epoch):
        real = self.sample_data["data"]
        # real
        pr, _ = np.histogram(real, bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0., 1, len(pr)), pr, label='real', color='g', linewidth=2)
        # pred
        emb, fake = [], []
        for _ in range(1):
            e, f = self.sess.run([self.embedding, self.pred_test], feed_dict={self.source:self.sample_data["data"]})
            emb.append(e)
            fake.append(f)
        pe, _ = np.histogram(np.concatenate(emb), bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0.,1,len(pe)), pe, label='embedding', color='b', linewidth=2)
        pf, _ = np.histogram(np.concatenate(fake), bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0.,1,len(pf)), pf, label='fake', color='r', linewidth=2)

        plt.legend()
        plt.title("epoch_{}".format(epoch))
        plt.savefig(os.path.join(self.fig_dir, "hist_epoch_{}.png".format(epoch)))
        plt.clf()
    
    def sample(self, epoch):
        real = self.sample_data["data"]
        fake, disc_code = self.sess.run([self.pred_test, self.dist_code_test], feed_dict={self.source:real})
        disc_code = np.argmax(disc_code, 1)
        for i in range(4):
            for j in range(2):
                idx = i*2 + j
                plt.subplot(4, 4, idx*2 + 1)
                plt.imshow(real[idx, :, :, 0], cmap=plt.cm.gray)
                plt.subplot(4, 4, idx*2 + 2)
                plt.imshow(fake[idx, :, :, 0], cmap=plt.cm.gray)
                plt.xticks([])
                plt.yticks([])
                plt.title(str(disc_code[idx]))
        plt.savefig(os.path.join(self.fig_dir, "sample_epoch_{}.png".format(epoch)))    
        plt.clf()    
    
    def train(self, epoches=1):
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data = self.data(self.batch_size, var_list=["data", "labels"])

                feed_dict = {self.source:data['data'], self.target:data['data'], self.labels:data['labels']}

                self.sess.run([self.reconstruct_solver, self.Q_solver], feed_dict=feed_dict)

                if cnt % 10 == 0:
                    r, q = self.sess.run([self.reconstruct_loss, self.Q_loss], feed_dict=feed_dict)
                    print ("Epoch [%3d/%3d] iter [%3d/%3d] rloss=%.3f qloss=%.3f" % (epoch, epoches, idx, batches_per_epoch, r, q))
            
            if epoch % 5 == 0:
                self.hist(epoch)
                self.sample(epoch)
        self.hist(epoch)
        self.sample(epoch)

        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)

class AutoEncoder(BasicTrainFramework):
    def __init__(self, batch_size, version="AE"):
        super(AutoEncoder, self).__init__(batch_size, version)
        
        self.data = datamanager_mnist(train_ratio=0.8, fold_k=None, expand_dim=True, norm=True)
        self.sample_data = self.data(self.batch_size, phase='test', var_list=["data", "labels"])

        self.emb_dim = 10

        self.encoder = CNN_Encoder(output_dim=self.emb_dim, sn=False)
        self.decoder = CNN_Decoder(sn=False)

        self.build_placeholder()
        self.build_network()
        self.build_optimizer()

        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        self.source = tf.compat.v1.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        self.target = tf.compat.v1.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
    
    def build_network(self):
        self.embedding = self.encoder(self.source, is_training=True, reuse=False)
        self.pred = self.decoder(self.embedding, is_training=True, reuse=False)
        self.pred_test = self.decoder(self.embedding, is_training=False, reuse=True)

        self.mean_pred, self.std_pred = tf.nn.moments(x=self.pred, axes=list(range(len(self.source.shape))))
        self.std_pred = tf.sqrt(self.std_pred)

    def build_optimizer(self):
        self.loss = mse(self.pred, self.target, self.batch_size)
        self.solver = tf.compat.v1.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.loss, var_list=self.encoder.vars + self.decoder.vars)
    
    def hist(self, epoch):
        real = self.sample_data["data"]
        # real
        pr, _ = np.histogram(real, bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0., 1, len(pr)), pr, label='real', color='g', linewidth=2)
        # pred
        emb, fake = [], []
        for _ in range(1):
            e, f = self.sess.run([self.embedding, self.pred_test], feed_dict={self.source:self.sample_data["data"]})
            emb.append(e)
            fake.append(f)
        pe, _ = np.histogram(np.concatenate(emb), bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0.,1,len(pe)), pe, label='embedding', color='b', linewidth=2)
        pf, _ = np.histogram(np.concatenate(fake), bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0.,1,len(pf)), pf, label='fake', color='r', linewidth=2)

        plt.legend()
        plt.title("epoch_{}".format(epoch))
        plt.savefig(os.path.join(self.fig_dir, "hist_epoch_{}.png".format(epoch)))
        plt.clf()
    
    def sample(self, epoch):
        real = self.sample_data["data"]
        fake = self.sess.run(self.pred_test, feed_dict={self.source:real})
        for i in range(4):
            for j in range(2):
                idx = i*2 + j
                plt.subplot(4, 4, idx*2 + 1)
                plt.imshow(real[idx, :, :, 0], cmap=plt.cm.gray)
                plt.subplot(4, 4, idx*2 + 2)
                plt.imshow(fake[idx, :, :, 0], cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.fig_dir, "sample_epoch_{}.png".format(epoch)))    
        plt.clf()    
    
    def train(self, epoches=1):
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data = self.data(self.batch_size, var_list=["data",])

                feed_dict = {self.source:data['data'], self.target:data['data']}

                self.sess.run(self.solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    print ("Epoch [%3d/%3d] iter [%3d/%3d] loss=%.3f" % (epoch, epoches, idx, batches_per_epoch, loss))
            mean, std = self.sess.run([self.mean_pred, self.std_pred], feed_dict=feed_dict)
            print ("mean=%.3f std=%.3f" % (mean, std))
            if epoch % 5 == 0:
                self.hist(epoch)
                self.sample(epoch)
        self.hist(epoch)
        self.sample(epoch)

        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)

def AE():
    # train AE
    ae = AutoEncoder(64)
    ae.train(epoches=10)

    # tsne
    ae.load_model()
    embs = []
    labels = []
    for i in range(100):
        data = ae.data(ae.batch_size, var_list=['data', 'labels'])
        emb = ae.sess.run(ae.embedding, feed_dict={ae.source:data['data']})
        embs.append(emb)
        labels.append(np.argmax(data['labels'], 1).reshape((ae.batch_size,)))
    embs = np.concatenate(embs, axis=0)
    labels = np.concatenate(labels)

    from sklearn.manifold import TSNE 

    model = TSNE(n_components=2, random_state=0)
    embs = model.fit_transform(embs)
    plt.scatter(embs[:, 0], embs[:, 1], c=labels)
    plt.colorbar() 

    plt.savefig(os.path.join(ae.fig_dir, "tsne.png"))

def AE_SUP():
    ae_sup = AutoEncoder_sup(64)
    ae_sup.train(epoches=10)

if __name__ == "__main__":
    AE()
    # AE_SUP()