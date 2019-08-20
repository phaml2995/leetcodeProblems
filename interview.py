

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
#--------------------------------------------------
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
#--------------------------------------------------
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
#--------------------------------------------------
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
#--------------------------------------------------
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
#--------------------------------------------------
#Reverse Integer
def reverse(x):
    if x < 0:  // 1st Bruteforce
        sign = -1
    else:
        sign = 1
    rever = int(str(abs(x))[::-1]) * sign
    if ( rever > ((2**31) -1) or rever < (-2**31)):
        return 0
    else:
        return rever
#--------------------------------------------------
    if x < 0: // 2nd Optimal
        sign = -1
    else:
        sign = 1

    final,remain = 0,abs(x)
    while remain:
        final = final * 10 + remain % 10
        remain //= 10
    final = final * sign
    if ( final > ((2**31) -1) or final < (-2**31)):
        return 0
    else:
        return final
#--------------------------------------------------
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
#--------------------------------------------------
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
#--------------------------------------------------
#Integer to Roman
"""
Given an integer, convert it into a roman numeral.
"""

class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        romanNum = ["I","IV","V","IX","X","XL","L","XC","C","CD","D","CM","M"]
        numeral =  [1,4,5,9,10,40,50,90,100,400,500,900,1000]
        final,i ="", len(numeral)-1

        if (num < 1) or (num > 3999):
            return 0

        while (num > 0):
            if (num - numeral[i] >= 0):
                final += romanNum[i]
                num -= numeral[i]
            else: i -= 1

        return final

#--------------------------------------------------
#Remove Dups from sorted List
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int: """Approach 1"""
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

     def removeDuplicates(self, nums: List[int]) -> int: """Approach 2"""
            if not nums:
                return 0

            newLength = 1
            for i in range(1,len(nums)):
                if nums[newLength-1] != nums[i]:
                    nums[newLength] = nums[i]
                    newLength+=1
            return newLength

#--------------------------------------------------
#Plus one
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if (len(digits) == 0):
            return 0

        strDigit = ''.join(str(i) for i in  digits)
        newDigits = [int(i) for i in str(int(strDigit)+1)]
        return newDigits
#--------------------------------------------------
#Length of the last word
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        string = s.split()
        if (len(string) == 0):
            return 0

        return len(string[-1])
#--------------------------------------------------
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
#--------------------------------------------------

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
#--------------------------------------------------
#Longest Substring without repeating
""" Given a string, find the length of the longest substring without repeating
    characters.
"""
    def lengthOfLongestSubstring(self, s: str) -> int: """Sliding window"""
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
#--------------------------------------------------
#Reverse a Linked List
""" Given a singly Linked list, reveres e it"""

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
        curr = headn
        nextNode = None

        while curr:
            nextNode = curr.next
            curr.next = prev
            prev = curr
            curr = nextNode
        return prev
#--------------------------------------------------
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
#--------------------------------------------------
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
#--------------------------------------------------
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
#--------------------------------------------------
"""Given a maximum number of characters in a line followed by a list of words,
I want to return a collection of strings where each line contains as many words
as possible concatenated by a space. The length of each string should not
exceed the maximum length.

There must be exactly one space between each word within each string of the output.
Each word will be composed of lowercase letters from the English alphabet.
There will be no punctuation.
The maximum length of each word can be assumed to be constant.
No single word will be longer than the given maximum length of characters in a line."""


def wrapLines(line_length,words):
    test_string = ""
    space = " "
    final = []   #list containing result strings
    i = 0        #used for iteration
    while (i < len(words)):
        if (len(test_string.strip()) + len(words[i]) + len(space)) > int(line_length):
            final.append(test_string.strip())

            test_string = ""

        else:
            test_string += words[i] + space
            i +=1
    if test_string:
        final.append(test_string.strip())
    for word in final:
        print(word)

words = ["abc","xyz","foobar","cuckoo","seven","hello"]
wrapLines("13",words)

#--------------------------------------------------
"""Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid."""


def isValid(s):
    """
    :type s: str
    :rtype: bool
    """
    openParen = ["(","[","{"]
    checkValid = []
    for paren in s:
        if paren in openParen:
            print(paren)
            checkValid.append(paren)
        elif (paren == ")" and (len(checkValid) != 0) and (checkValid[-1] == "(")):
            print(paren)
            checkValid.pop()
        elif (paren == "]" and (len(checkValid) != 0) and (checkValid[-1] == "[")):
            checkValid.pop()
        elif (paren == "}" and (len(checkValid) != 0) and (checkValid[-1c] == "{") ):
            print(checkValid[0])
            print(paren)
            checkValid.pop()
        else:
            return False
    return (len(checkValid) == 0)
#--------------------------------------------------

"""Best time to buy and sell stock"""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        buy_price = max(prices)
        final = 0
        for i in range(len(prices)):
            if prices[i] < buy_price:
                buy_price = prices[i]
            else:
                final = max(final,prices[i] - buy_price)


        return final
#--------------------------------------------------
"""Given a paragraph and a list of banned words, return the most frequent word
that is not in the list of banned words.  It is guaranteed there is at least
one word that isn't banned, and that the answer is unique.
Words in the list of banned words are given in lowercase, and free of
punctuation.  Words in the paragraph are not case sensitive.The answer is in lowercase. """


def word(para,banned):

    punct = "!?',;.:"
    for c in punct:
        para = para.replace(c, " ")
    wordList = para.lower().split()

    print (wordList)
    wordVal = {}
    res = ""
    for word in wordList:
        if word not in banned:
            if word in wordVal:
                wordVal[word] +=1
            else:
                wordVal[word] = 1

    for key in wordVal.keys():
        if (res == "") or (wordVal[key] > wordVal[res]):
            res = key

    return res
#--------------------------------------------------
""" Unique Binary Search Tree: Given n, how many structurally
unique BST's (binary search trees) that store values 1 ... n?"""

class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0] * (n+1)
        dp[0] = 1
        dp[1] = 1

        for i in range(2,n+1):
            for j in range(1,i+1):
                dp[i] += dp[j-1] * dp[i-j]

        return dp[n]
#--------------------------------------------------
"""Valid Palindrome"""
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return True

        i,j = 0,len(s) - 1
        while i < j:
            if not s[i].isalnum():
                i += 1
            elif not s[j].isalnum():
                j -= 1

            else:
                if ( s[i].lower() == s[j].lower()):
                    i+= 1
                    j-=1
                else:
                    return False
        return True
#--------------------------------------------------
"""Reverse Wprds In a String"""
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        new = s.strip().split()
        i,j = 0,len(new)-1
        while i < j:
            new[i],new[j] = new[j],new[i]
            i+=1
            j-=1
        return ' '.join(new)

#--------------------------------------------------
""""Symmetric Tree"""
 Definition for a binary tree node.
 class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
import collections

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
         if not root:    # BFS
             return True

         queue = collections.deque([(root,root)])

         while queue:
             currNode1,currNode2 = queue.popleft()
             if not currNode1 and not currNode2:
                 continue
             elif not currNode1 or not currNode2:
                 return  False
             elif currNode1.val != currNode2.val:
                 return False
             queue.append((currNode1.left,currNode2.right))
             queue.append((currNode1.right,currNode2.left))

         return True

        def checksymmetric(treeLeft,treeRight): #DFS
            if not treeLeft and not treeRight:
                return True
            elif treeLeft and treeRight:
                return (treeLeft.val == treeRight.val
                        and checksymmetric(treeLeft.left,treeRight.right)
                        and checksymmetric(treeLeft.right,treeRight.left))
            return False

        return not root or checksymmetric(root.left,root.right)
#--------------------------------------------------

"""Kth Largest Element in an array"""

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heap = nums[:k]
        heapq.heapify(heap)
        for num in nums[k:]:
            if num > heap[0]:
                heapq.heappushpop(heap,num)

        return heap[0]
#--------------------------------------------------

"""Subsets"""

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        final = [[]]
        for n in nums:
            for i in range(len(final)):
                final.append(final[i]+[n])
        return final

#--------------------------------------------------
"""Find Peak Element"""

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) <= 1:  """Bruteforce approach"""
            return 0
        final = 0
        for i in range(len(nums)-1):
            if i + 2 == len(nums):
                if nums[i+1] > nums[i]:
                    return i+1
            if  nums[i] > nums[i+1] and nums[i] > nums[i-1]:
                return i
#****************************************************
        l,r = 0,len(nums)-1    """Binary Seach"""
        while l < r:
            mid = l + (r-l)//2
            curr = nums[mid]
            if (curr > nums[mid+1]):
                r = mid
            else:
                l = mid + 1
        return l
#--------------------------------------------------
"""Permutations"""

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:


        def get_perm(i):
            if i == len(nums) - 1:
                final.append(nums[:])
                return

            for j in range(i,len(nums)):
                nums[i],nums[j] = nums[j],nums[i]
                get_perm(i+1)
                nums[i],nums[j] = nums[j],nums[i]



        final = []
        get_perm(0)
        return final
#--------------------------------------------------
"""Find All Duplicates in an array"""

def findDuplicates(self, nums: List[int]) -> List[int]:
        dic = {}
        final = []
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1

        for key in dic.keys():
            if dic[key] > 1:
                final.append(key)


def findDuplicates(self, nums: List[int]) -> List[int]:

        count = 0
        nums = sorted(nums)

        for i in range(1,len(nums)):
            if nums[i] == nums[i-1]:
                nums[i],nums[count] = nums[count],nums[i]
                count +=1


        return nums[:count]
#--------------------------------------------------

"""Count and Say"""

class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        s = "1"
        # for _ in range(n-1):
        #     s = ''.join(str(len(list(group))) + key for key,group in itertools.groupby(s))
        # return s

        for _ in range(n-1):
            s = self.next_num(s)
        return s

    def next_num(self,s):
        final,i = "",0
        while i < len(s):
            count = 1
            while i + 1 < len(s) and s[i] == s[i+1]:
                i+= 1
                count +=1
            final += str(count) + s[i]
            i+=1
        return final

#--------------------------------------------------
"""Group Anagrams"""

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = {}
        for word in strs:
            hsh = 1
            for char in word:
                hsh *= hash(char)
            if hsh in dic:
                dic[hsh].append(word)
            else:
                dic[hsh] = [word]

        final = []
        for key in dic.keys():
            final.append(dic[key])

        return final
#--------------------------------------------------

"""ZigZag Conversion"""
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows > len(s):
            return s

        final = [""]*numRows
        index,step = 0,1

        for char in s:
            final[index] += char
            if index == 0:
                step = 1
            if index == numRows -1:
                step = -1
            index += step

        return ''.join(final)
