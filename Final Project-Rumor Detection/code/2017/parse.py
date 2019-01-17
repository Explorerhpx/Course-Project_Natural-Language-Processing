import json
import math
import os
import random


hash_hat_function = {}  ##Record whether a and b has between caculated


def alreadyCaculate(node1, node2):

    step1 = hash_hat_function.get(node1['mid'])
    if not step1:
        return False

    step2 = step1.get(node2['mid'])
    if not step2:
        return False

    return step2


def markCaculate(node1, node2, ans):
    
    if not hash_hat_function.get(node1['mid']):
        hash_hat_function[node1['mid']] = {}
    if not hash_hat_function.get(node2['mid']):
        hash_hat_function[node2['mid']] = {}

    hash_hat_function[node1['mid']][node2['mid']] = ans
    hash_hat_function[node2['mid']][node1['mid']] = ans


class Tree:
    def __init__(self, filename):
        self.hashtree = {}
        with open(filename) as fp:
            self.list = json.loads(fp.read())
        self.constructTree()

        self.root = self.list[0]

    def constructTree(self):
        ##Build Hash Tree
        for item in self.list:
            self.hashtree[item['mid']] = item
            self.hashtree[item['mid']]['children'] = []
        ##Build connection of parent and child
        for item in self.list:
            if self.hashtree.get(item['parent']):
                self.hashtree[item['parent']]['children'].append(item)


    def convertToJson(self,path,filename):
        with open(os.path.join(path,'tree'+filename), 'w') as fp:
            dump = {
                'root': self.root,
                'hashtree': self.hashtree
            }
            fp.write(json.dumps(dump))


    def extractUserFeature(self, node):
        length_of_description = len(node['user_description'])
        followers_count = node['followers_count'] / 100
        verified = int((node['verified'] == True)) * 10
        picture = int((node['picture'] == None)) * 10
        gender = int((node['gender'] == 'f')) * 10

        return [length_of_description, followers_count, verified, picture, gender]

    def Ngram(self, text):
        ngram = set()

        for i in range(len(text)):
            ngram.add(text[i])

        for j in range(len(text) - 1):
            ngram.add(text[j] + text[j + 1])

        return ngram

    def isLeafNode(self, node, tree):
        node_id = node['mid']
        if tree.hashtree[node_id]['children'] == []:
            return True
        else:
            return False

    def Jaccord(self, set_i, set_j):
        union = len(set_i | set_j)
        cross = len(set_i & set_j)
        if union == 0:
            return 0
        else:
            return cross / union

    def calculateJaccord(self, node1, node2):
        set1 = self.Ngram(node1['text'])
        set2 = self.Ngram(node2['text'])

        return self.Jaccord(set1, set2)

    def f(self, node1, node2):
        alpha = 0.99
        t1 = node1['t']
        t2 = node2['t']
        u1 = self.extractUserFeature(node1)
        u2 = self.extractUserFeature(node2)
        J = self.calculateJaccord(node1, node2)

        step1 = math.exp(-abs(t1 - t2)/100000)
        step2 = alpha * self.Sig(u1, u2)/10
        step3 = (1 - alpha) * J
        ans = step1 * (step2 + step3)
        return ans

    def Sig(self, vector1, vector2):
        sum = 0
        for i in range(len(vector1)):
            sum += (vector1[i]-vector2[i])**2
        return math.sqrt(sum)

    def PTK(self, another_tree):
        sum = 0
        for node in self.list:
            couterpart = self.findProperNode(node, another_tree)
            temp = self.hat(node, couterpart, another_tree)
            sum += temp
        return sum

    def cPTK(self, another_tree):
        sum = self.PTK(another_tree)
        for node in self.list:
            couterpart = self.findProperNode(node, another_tree)

            ancestors1 = self.findAncestors(node)
            ancestors2 = self.findAncestors(couterpart)
            length = min(len(ancestors1),len(ancestors2))
            if length == 0:
                continue

            for index in range(length):
                sum += self.hat(ancestors1[index], ancestors2[index], another_tree)

        return sum  

    def findAncestors(self, node):
        ancestors = []

        while(node):
            parent = node.get('parent')
            if parent:
                node = self.hashtree.get(parent)
                ancestors.append(node)
            else:
                break

        return ancestors

    def compareToAnotherTree(self, method, another_tree):
        if method == 'PTK':
            sum = self.PTK(another_tree)
            return sum
        if method == 'cPTK':
            sum = self.cPTK(another_tree)
            return sum

    def hat(self, node, couterpart, another_tree):

        ##Judge whether has been caculated before
        if (not node) or (not couterpart):
            return 0 

        ans = alreadyCaculate(node, couterpart)
        if ans:
            return ans

        try:
            ##Recursive Base
            if self.isLeafNode(node, self):
                ans = self.f(node, couterpart)

            else:
                upper = min(self.numberOfChild(node, self), self.numberOfChild(couterpart, another_tree))
                ans = 1
                for k in range(upper):
                    ans += math.log((1 + self.hat(self.kChild(node, k, self), self.kChild(couterpart, k, another_tree), another_tree)))

                ans = math.exp(ans) * self.f(node, couterpart)
        except:
            print('Overflow')
            ans = 1

        ##mark as caculated
        markCaculate(node, couterpart, ans)
        return ans

    def numberOfChild(self, node, tree):
        return len(tree.hashtree[node['mid']]['children'])

    def kChild(self, node, k, tree):
        return tree.hashtree[node['mid']]['children'][k]

    def findProperNode(self, node, another_tree):
        maxf = -100000
        maxnode = another_tree.list[0]
        for item in another_tree.list:
            tempf = self.f(node, item)
            if tempf < 0.1:
                return maxnode

            if tempf > maxf:
                maxnode = item
                maxf = tempf

        return maxnode


def main():
    forest = []  ##set of trees
    data_path = 'data'
    tree_path = 'tree'
    result_path = 'result'
    files = os.listdir(data_path)
    size = 20


    ##For SVM
    cost_matrix = [[0 for i in range(len(files))] for j in range(len(files))]  ##Cost Matirx
    result_list = [0 for i in range(len(files))]

    for (index, filename) in enumerate(files): 
        tree = Tree(os.path.join(data_path, filename))
        tree.convertToJson(tree_path, filename)
        forest.append(tree)
 

    for outer_index in range(0, len(forest)):
        for inner_index in range(outer_index + 1, len(forest)):
            ##Avoid compare twice for each pair of node
            Hat1 = forest[outer_index].compareToAnotherTree('PTK', forest[inner_index])
            Hat2 = forest[inner_index].compareToAnotherTree('PTK', forest[outer_index])
            cost_matrix[outer_index][inner_index] = Hat1 + Hat2
            cost_matrix[inner_index][outer_index] = Hat1 + Hat2
            print("inner",inner_index)
        print("outer",outer_index)

        with open(os.path.join(result_path,str(outer_index)+'result.json'),'w') as fp:
            fp.write(json.dumps(cost_matrix))


if __name__ == '__main__':
    main()
