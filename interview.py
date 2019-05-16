

# Anogram Check
def anogram(s,t):
    s = s.replace(" ",'').lower()
    t = t.replace(" ",'').lower()

    l = {}
    if (len(s) != len(t)):
        return False

    for letter in s:
        if letter in l:
            l[letter] += 1
        else:
            l[letter] = 1

    for letter in t:
        if letter in l:
            l[letter] -= 1
        else:
            return False

    for char in l.keys():
        if l[char] != 0:
            return False

    return True

if(anogram("public relations","crap built on lies")):
    print("It's an anogram!")
else:
    print("Not an anogram!")

#Pair Sum
def pair_sum(arr,k):
    """
    Given an array of integers, return indices of the two numbers such that
    they add up to a specific target.You may assume that each input would have
    exactly one solution, and you may not use the same element twice.
    """
     def twoSum(self, nums: List[int], target: int) -> List[int]:
        newList = {};
        for i,num in enumerate(nums):
            result = target - num
            if result in newList:
                return [i, newList[result]]
            newList[num] = i
        return newList

#Find missing Elements
#1st solutioj
def finder(a1,a2):
    sum1 = sum(a1)
    sum2 = sum(a2)
    missing_el = sum1 - sum2
    return missing_el

print(finder([1,2,3,4,5,6,7],[3,7,2,1,4,6]))
#second solution
import collections
def finder2(a1,a2):
    d = collections.defaultdict(int)
    for num in a2:
        d[num] += 1
    for num in a1:
        if d[num] == 0:
            return num
        else:
            d[num] -= 1

##Merge Names
def unique_names(names1,names2):
    both_names = names1 + names2
    names_set = set()
    final = []
    for name in both_names:
        if name not in names_set:
            final.append(name)
            names_set.add(name)
    return final
names1 = ["Ava", "Emma", "Olivia"]
names2 = ["Olivia", "Sophia", "Emma"]
print(unique_names(names1,names2))

##Palindrome
def Palindrome(string):
    reversed_w = "".join(reversed(string)).lower()
    if string.lower() == reversed_w:
        return True
    else:
        return False

word = "Mom"
print (Palindrome(word))

def sum_func(n):
    if len(str(n)) == 1:
        return n
    else:
        return n%10 + sum_func(n/10)

print(sum_func(43))

#Reverse Integer
def reverse(x):
    if x < 0:
        sign = -1
    else:
        sign = 1
    rever = int(str(abs(x))[::-1]) * sign
    if ( rever > ((2**31) -1) or rever < (-2**31)):
        return 0
    else:
        return rever

#Palindrome number
class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        neX = str(x)[::-1]
        if (neX[-1] == '-' or neX != str(x)):
            return False
        else:
            return True

#Roman to Integer
class Solution:
    def romanToInt(self, s: str) -> int:
        romVal = {'I':1, 'V':5, 'X':10, 'L':50, 'C': 100, 'D':500, 'M': 1000}
        finalVal = 0
        prevVal = 0

        for i in range(len(s)):
            curVal = romVal[s[i]]
            if curVal > prevVal:
                finalVal += curVal - prevVal * 2
            else:
                finalVal += curVal

            prevVal = curVal

        return finalVal

#Remove Dups from sorted List
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 0:
            return 0
        i = 0
        temp = nums[i];
        newLength = 1;
        for item in nums[1:]:
            if (item != temp):
                temp = item
                nums[newLength] = item
                newLength += 1
                i += 1
        nums = nums[:newLength]
        return newLength

#Plus one
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if (len(digits) == 0):
            return 0

        strDigit = ''.join(str(i) for i in  digits)
        newDigits = [int(i) for i in str(int(strDigit)+1)]
        return newDigits

#Length of the last word
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        string = s.split()
        if (len(string) == 0):
            return 0

        return len(string[-1])

#Reverse string
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        l,r = 0, len(s)-1
        while l < r:
            s[l],s[r] = s[r],s[l]
            l+= 1
            r -= 1


#Container with most Water
'''Given n non-negative integers a1, a2, ..., an , where each represents a point at
coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line
i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a
container, such that the container contains the most water.'''

 def maxArea(self, height: List[int]) -> int:
        left, right, width, final = 0, len(height) - 1, len(height) - 1, 0

        for i in range(width,0,-1):
            if (height[left] < height[right]):
                final,left = max(final,height[left]*i), left + 1
            else:
                final,right = max(final,height[right]*i), right - 1

        return final

#Longest Substring without repeating
""" Given a string, find the length of the longest substring without repeating
    characters.
"""
    def lengthOfLongestSubstring(self, s: str) -> int:
        if (len(s) == 0):
            return 0
        maxLength = 0
        seen = set()

        length,i,j = len(s),0,0

        while i < length and j < length:
            if s[i] in seen:
                seen.remove(s[j])
                j += 1
            else:
                seen.add(s[i])
                maxLength = max(maxLength, len(seen))
                i += 1
        return maxLength

#Reverse a Linked List
""" Given a singly Linked list, reverese it"""

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        curr = head
        nextNode = None

        while curr:
            nextNode = curr.next
            curr.next = prev
            prev = curr
            curr = nextNode
        return prev

# Remove Nth node From end of List
"""Given a linked list, remove the n-th node from the end of
list and return its head."""

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        marker1 = head  """markers to traverse the list"""
        marker2 = head
        for i in range (n):
            if not marker2.next:
                head = head.next
                return head
            marker2 = marker2.next

        while marker2.next:
            marker1 = marker1.next
            marker2 = marker2.next
        marker1.next = marker1.next.next """set the next node to the last node"""
        return head

#Add Two number
"""You are given two non-empty linked lists representing two non-negative
integers. The digits are stored in reverse order and each of their nodes
contain a single digit. Add the two numbers and return it as a linked list.
"""

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        temp = 0
        curr = newNode = ListNode(0)
        while l1 or l2 or temp:

            if l1 is not None:
                temp += l1.val
                l1 = l1.next

            if l2 is not None:
                temp += l2.val
                l2 = l2.next

            temp,val = divmod(temp,10)
            curr.next = ListNode(val)
            curr = curr.next
        return newNode.next

#Merge Two sorted List: Using Linked List
""" Merge Two sorted Linked lists and return in as a new list. The new list
should be made by slicing together the nodes of the first two lists.
"""

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        newNode = curr = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        curr.next = l1 or l2
        return newNode.next
