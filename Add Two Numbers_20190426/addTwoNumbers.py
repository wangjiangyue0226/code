#-*- coding:utf-8 -*-
class ListNode:
    """单链表的结点"""
    def __init__(self, val):
        # val存放数据元素
        self.val = val
        # _next是下一个节点的标识
        self.next = None

class MyLinkedList(object):
    """单链表"""
    def __init__(self):
        self._head = None
        self.length = 0

    def is_empty(self):
        """判断链表是否为空"""
        return self._head == None

    def get_head(self):
        return self._head
    
    def addAtHead(self, val):
        """头部添加元素"""
        # 先创建一个保存val值的节点
        node = ListNode(val)
        # 将新节点的链接域next指向头节点，即_head指向的位置
        node.next = self._head
        # 将链表的头_head指向新节点
        self._head = node
        self.length += 1

    def addAtTail(self, val):
        """尾部添加元素"""
        node = ListNode(val)
        # 先判断链表是否为空，若是空链表，则将_head指向新节点
        if self.is_empty():
            self._head = node
        # 若不为空，则找到尾部，将尾节点的next指向新节点
        else:
            cur = self._head
            while cur.next:
                cur = cur.next
            cur.next = node
        self.length += 1

    def addAtIndex(self, index, val):
        """指定位置添加元素"""
        # 若指定位置index为第一个元素之前，则执行头部插入
        if index <= 0:
            self.addAtHead(val)
        # 若指定位置超过链表尾部，则执行尾部插入
        elif index > (self.length - 1):
            self.addAtTail(val)
        # 找到指定位置
        else:
            node = ListNode(val)
            count = 0
            # pre用来指向指定位置pos的前一个位置pos-1，初始从头节点开始移动到指定位置
            pre = self._head
            while count < (index - 1):
                count += 1
                pre = pre.next
            # 先将新节点node的next指向插入位置的节点
            node.next = pre.next
            # 将插入位置的前一个节点的next指向新节点
            pre.next = node
            self.length += 1

    def deleteAtIndex(self, index):
        """
        删除指定位置的元素
        """
        node = ListNode(0)
        if index <= 0 or index >= self.length:
            return -1
        else:
            cur = self._head
            pre = None
            cnt = 0
            while cnt < index:
                pre = cur
                cnt += 1
                cur = cur.next
            pre.next = cur.next
            self.length -= 1

    def search(self, item):
        """链表查找节点是否存在，并返回True或者False"""
        cur = self._head
        while cur != None:
            if cur.item == item:
                return True
            cur = cur.next
        return False

    def travel(self):
        """遍历链表"""
        cur = self._head
        while cur != None:
            print(cur.val)
            cur = cur.next
        print("")

class Solution:
    def addTwoNumbers(self,l1,l2):
        '''
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        '''
        head = temp = ListNode(0)
        carry = 0

        while l1 or l2 or carry:
            temp1 = l1.val if l1 else 0
            temp2 = l2.val if l2 else 0
            tempSum = temp1 + temp2 + carry

            temp.next = ListNode(tempSum % 10)
            temp = temp.next
            carry = tempSum // 10

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        cur = head.next
        while(cur!=None):
            print(cur.val)
            cur=cur.next
        return head.next

if __name__ == '__main__':
    #链表1:4->4->4
    l1=MyLinkedList()
    l1.addAtHead(4)
    l1.addAtIndex(1,4)
    l1.addAtIndex(2,4)
    l1.travel()

    # 链表2:6->6->9
    l2 = MyLinkedList()
    l2.addAtHead(6)
    l2.addAtIndex(1, 6)
    l2.addAtIndex(2, 9)
    l2.travel()

    s = Solution()
    s.addTwoNumbers(l1.get_head(),l2.get_head())
