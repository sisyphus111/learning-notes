#!/usr/bin/env python3
"""Check Python example syntax and run representative cases in Python 3.10+."""
import ast
import contextlib
import io
import re
import sys
from pathlib import Path
from typing import Optional, List

ROOT = Path(__file__).resolve().parents[1]

class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

class Node:
    def __init__(self, val):
        self.val, self.next, self.random = val, None, None

CASES = {
 'fib': 'assert s.fib(0) == 0 and s.fib(10) == 55',
 'uniquePaths': 'assert s.uniquePaths(3, 7) == 28 and s.uniquePaths(1, 1) == 1',
 'uniquePathsWithObstacles': 'assert s.uniquePathsWithObstacles([[0,0,0],[0,1,0],[0,0,0]]) == 2',
 'minPathSum': 'assert s.minPathSum([[1,3,1],[1,5,1],[4,2,1]]) == 7',
 'isAnagram': 'assert s.isAnagram("anagram","nagaram") and not s.isAnagram("rat","car")',
 'groupAnagrams': 'assert sorted(map(len, s.groupAnagrams(["eat","tea","tan","ate","nat","bat"]))) == [1,2,3]',
 'longestConsecutive': 'assert s.longestConsecutive([100,4,200,1,3,2]) == 4',
 'firstMissingPositive': 'assert s.firstMissingPositive([3,4,-1,1]) == 2',
 'threeSum': 'assert sorted(s.threeSum([-1,0,1,2,-1,-4])) == [[-1,-1,2],[-1,0,1]]',
 'maxArea': 'assert s.maxArea([1,8,6,2,5,4,8,3,7]) == 49',
 'trap': 'assert s.trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6',
 'sortColors': 'a=[2,0,2,1,1,0]; s.sortColors(a); assert a == [0,0,1,1,2,2]',
 'nextPermutation': 'a=[3,2,1]; s.nextPermutation(a); assert a == [1,2,3]',
 'moveZeroes': 'a=[0,1,0,3,12]; s.moveZeroes(a); assert a == [1,3,12,0,0]',
 'findDuplicate': 'assert s.findDuplicate([1,3,4,2,2]) == 2',
 'minSubArrayLen': 'assert s.minSubArrayLen(7,[2,3,1,2,4,3]) == 2',
 'numSubarrayProductLessThanK': 'assert s.numSubarrayProductLessThanK([10,5,2,6],100) == 8',
 'lengthOfLongestSubstring': 'assert s.lengthOfLongestSubstring("abcabcbb") == 3',
 'findAnagrams': 'assert s.findAnagrams("cbaebabacd","abc") == [0,6]; assert s.findAnagrams("ab","abc") == []',
 'minWindow': 'assert s.minWindow("ADOBECODEBANC","ABC") == "BANC"',
 'countSubarrays': 'assert s.countSubarrays([1,1,2,2],2,2) == 1',
 'findSubstring': 'assert sorted(s.findSubstring("barfoothefoobarman",["foo","bar"])) == [0,9]',
 'subarraySum': 'assert s.subarraySum([0,0,0],0) == 6',
 'productExceptSelf': 'assert s.productExceptSelf([-1,1,0,-3,3]) == [0,0,9,0,0]',
 'search': 'assert s.search([1,3,5,7],5) == 2; assert s.search([1,3,5,7],2) == -1',
 'searchRange': 'assert s.searchRange([5,7,7,8,8,10],8) == [3,4]',
 'findPeakElement': 'assert s.findPeakElement([1,2,3,1]) == 2',
 'findMin': 'assert s.findMin([3,4,5,1,2]) == 1',
 'countElements': 'assert s.countElements([3,1,2],1) == 2',
 'findMedianSortedArrays': 'assert s.findMedianSortedArrays([1,3],[2]) == 2',
 'findTargetSumWays': 'assert s.findTargetSumWays([1,1,1,1,1],3) == 5; assert s.findTargetSumWays([0,0],0) == 4',
 'canPartition': 'assert s.canPartition([1,5,11,5]); assert not s.canPartition([1,2,3,5])',
 'lengthOfLongestSubsequence': 'assert s.lengthOfLongestSubsequence([1,2,3,4,5],9) == 3',
 'findMaxForm': 'assert s.findMaxForm(["10","0001","111001","1","0"],5,3) == 4',
 'coinChange': 'assert s.coinChange([1,2,5],11) == 3; assert s.coinChange([2],3) == -1',
 'numSquares': 'assert s.numSquares(12) == 3 and s.numSquares(13) == 2',
 'change': 'assert s.change(5,[1,2,5]) == 4',
 'longestCommonSubsequence': 'assert s.longestCommonSubsequence("abcde","ace") == 3',
 'minDistance': 'assert s.minDistance("horse","ros") == 3; assert s.minDistance("","abc") == 3; assert s.minDistance("abc","") == 3',
 'lengthOfLIS': 'assert s.lengthOfLIS([10,9,2,5,3,7,101,18]) == 4',
 'maxSubArray': 'assert s.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]) == 6',
 'maxProduct': 'assert s.maxProduct([2,3,-2,4]) == 6',
 'maxSubarraySumCircular': 'assert s.maxSubarraySumCircular([5,-3,5]) == 10',
 'wordBreak': 'assert s.wordBreak("leetcode",["leet","code"]); assert not s.wordBreak("b",["a"]); assert not s.wordBreak("a",["aaa"])',
 'longestValidParentheses': 'assert s.longestValidParentheses(")()())") == 4',
 'isInterleave': 'assert s.isInterleave("aabcc","dbbca","aadbbcbcac"); assert not s.isInterleave("aabcc","dbbca","aadbbbaccc")',
 'maxScore': 'assert s.maxScore([-1,4,5,-2],[-5,-1,-3,-2,-4]) == -1',
 'numDistinct': 'assert s.numDistinct("rabbbit","rabbit") == 3',
 'longestPalindromeSubseq': 'assert s.longestPalindromeSubseq("bbbab") == 4',
 'minScoreTriangulation': 'assert s.minScoreTriangulation([1,3,1,4,1,5]) == 13',
 'longestPalindrome': 'assert s.longestPalindrome("babad") in ("bab","aba")',
 'maxCoins': 'assert s.maxCoins([3,1,5,8]) == 167',
 'stoneGame': 'assert s.stoneGame([5,3,4,5])',
 'minCostConnectPoints': 'assert s.minCostConnectPoints([[0,0],[2,2],[3,10],[5,2],[7,0]]) == 20',
 'networkDelayTime': 'assert s.networkDelayTime([[2,1,1],[2,3,1],[3,4,1]],4,2) == 2',
 'orangesRotting': 'assert s.orangesRotting([[2,1,1],[1,1,0],[0,1,1]]) == 4; assert s.orangesRotting([[0]]) == 0',
 'minMutation': 'assert s.minMutation("AACCGGTT","AACCGGTA",["AACCGGTA"]) == 1',
 'ladderLength': 'assert s.ladderLength("hit","cog",["hot","dot","dog","lot","log","cog"]) == 5',
 'canFinish': 'assert s.canFinish(2,[[1,0]]); assert not s.canFinish(2,[[1,0],[0,1]])',
 'findOrder': 'assert s.findOrder(2,[[1,0]]) == [0,1]',
 'canJump': 'assert s.canJump([2,3,1,1,4]); assert not s.canJump([3,2,1,0,4])',
 'jump': 'assert s.jump([2,3,1,1,4]) == 2',
 'partitionLabels': 'assert s.partitionLabels("ababcbacadefegdehijhklij") == [9,7,8]',
 'rangeBitwiseAnd': 'assert s.rangeBitwiseAnd(5,7) == 4',
 'findKthLargest': 'assert s.findKthLargest([3,2,3,1,2,4,5,5,6],4) == 4',
 'kSmallestPairs': 'assert s.kSmallestPairs([1,7,11],[2,4,6],3) == [[1,2],[1,4],[1,6]]; assert s.kSmallestPairs([1],[2],5) == [[1,2]]',
 'dailyTemperatures': 'assert s.dailyTemperatures([73,74,75,71,69,72,76,73]) == [1,1,4,2,1,1,0,0]',
 'largestRectangleArea': 'assert s.largestRectangleArea([2,1,5,6,2,3]) == 10',
 'maxSlidingWindow': 'assert s.maxSlidingWindow([1,3,-1,-3,5,3,6,7],3) == [3,3,5,5,6,7]',
 'numOfUnplacedFruits': 'assert s.numOfUnplacedFruits([4,2,5],[3,5,4]) == 1',
 'combinationSum': 'assert sorted(s.combinationSum([2,3,6,7],7)) == [[2,2,3],[7]]',
 'permute': 'assert len({tuple(x) for x in s.permute([1,2,3])}) == 6',
 'letterCombinations': 'assert len(s.letterCombinations("23")) == 9; assert s.letterCombinations("") == []',
 'combine': 'assert len(s.combine(4,2)) == 6',
 'combinationSum3': 'assert s.combinationSum3(3,7) == [[1,2,4]]',
 'generateParenthesis': 'assert len(s.generateParenthesis(3)) == 5',
 'subsets': 'assert len(s.subsets([1,2,3])) == 8',
 'partition': 'assert s.partition("aab") == [["a","a","b"],["aa","b"]]',
 'solveNQueens': 'assert len(s.solveNQueens(4)) == 2',
 'maxDepth': 'assert s.maxDepth(TreeNode(1,TreeNode(2))) == 2',
 'isBalanced': 'assert s.isBalanced(TreeNode(1,TreeNode(2))); assert not s.isBalanced(TreeNode(1,TreeNode(2,TreeNode(3))))',
 'rightSideView': 'assert s.rightSideView(TreeNode(1,TreeNode(2),TreeNode(3))) == [1,3]',
 'isValidBST': 'assert s.isValidBST(TreeNode(2,TreeNode(1),TreeNode(3)))',
 'levelOrder': 'assert s.levelOrder(TreeNode(2,TreeNode(1),TreeNode(3))) == [[2],[1,3]]',
 'zigzagLevelOrder': 'assert s.zigzagLevelOrder(TreeNode(2,TreeNode(1),TreeNode(3))) == [[2],[3,1]]',
 'findBottomLeftValue': 'assert s.findBottomLeftValue(TreeNode(2,TreeNode(1),TreeNode(3))) == 1',
 'reverseList': 'a=ListNode(1,ListNode(2)); b=a.next; assert s.reverseList(a) is b and b.next is a and a.next is None',
 'reverseKGroup': 'a=ListNode(1,ListNode(2,ListNode(3))); b=a.next; assert s.reverseKGroup(a,2) is b and b.next is a and a.next.val == 3',
 'isPalindrome': 'assert s.isPalindrome(ListNode(1,ListNode(2,ListNode(2,ListNode(1)))))',
 'sortList': 'r=s.sortList(ListNode(4,ListNode(2,ListNode(1,ListNode(3))))); values=[]\nwhile r: values.append(r.val); r=r.next\nassert values == [1,2,3,4]',
}

def main():
    if sys.version_info < (3,10):
        raise SystemExit('Behavior checks require Python 3.10+ for Counter comparisons.')
    syntax = behavior = reference_examples = contextual_examples = 0
    errors = []
    for path in sorted(ROOT.glob('*/*.md')):
        if path.parent.name == 'STL': continue
        for index, match in enumerate(re.finditer(r'^```python\n(.*?)^```',path.read_text(),re.M|re.S)):
            code = match[1]
            try:
                tree = ast.parse(code)
                syntax += 1
                if path.parent.name == 'Python':
                    # Multi-file projects are syntax-checked here; they need their
                    # documented files and build environment for execution.
                    if code.startswith('# 文件：'):
                        contextual_examples += 1
                        continue
                    expected = re.findall(r'# 输出：(.*)', code)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        exec(compile(tree, str(path), 'exec'), {})
                    assert output.getvalue().splitlines() == expected, (output.getvalue(), expected)
                    reference_examples += 1
                    continue
                solution = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Solution'), None)
                methods = {node.name for node in solution.body if isinstance(node, ast.FunctionDef)} if solution else set()
                complete = solution is not None
                test = None
                for method, case in CASES.items():
                    if method in methods: test = case; break
                if 'def singleNumber(' in code:
                    test = 'assert s.singleNumber([2,2,3,2]) == 3' if 'range(32)' in code else 'assert s.singleNumber([4,1,2,1,2]) == 4'
                if 'def rob(' in code:
                    test = 'assert s.rob([1,2,3,1]) == 4' if 'List[int]' in code else 'assert s.rob(TreeNode(3,TreeNode(2),TreeNode(3))) == 5'
                if 'def maxProfit(' in code:
                    test = 'assert s.maxProfit(2,[3,2,6,5,0,3]) == 7' if 'k: int' in code else ('assert s.maxProfit([1,2,3,0,2]) == 3' if 'dp[i - 2]' in code else 'assert s.maxProfit([7,1,5,3,6,4]) == 7')
                if 'def mergeStones(' in code:
                    test = 'assert s.mergeStones([3,2,4,1],2) == 20' if 'k: int' in code else 'assert s.mergeStones([3,2,4,1]) == 20'
                class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
                class_tests = {
                    'Trie': 't=Trie(); t.insert("apple"); assert t.search("apple") and not t.search("app"); assert t.startsWith("app")' if 'def startsWith' in code else 't=Trie(); t.insert("apple"); assert t.search("apple") and not t.search("app")',
                    'NumArray': 'a=NumArray([1,3,5]); assert a.sumRange(0,2)==9; a.update(1,2); assert a.sumRange(0,2)==8',
                    'LRUCache': 'l=LRUCache(2); l.put(1,1); l.put(2,2); assert l.get(1)==1; l.put(3,3); assert l.get(2)==-1',
                    'MedianFinder': 'm=MedianFinder(); m.addNum(1); m.addNum(2); assert m.findMedian()==1.5; m.addNum(3); assert m.findMedian()==2',
                    'MinStack': 'm=MinStack(); m.push(-2); m.push(0); m.push(-3); assert m.getMin()==-3; m.pop(); assert m.top()==0 and m.getMin()==-2',
                }
                for name, case in class_tests.items():
                    if name in class_names:
                        test, complete = case, True
                        break
                if not complete or not test: continue
                namespace = dict(List=List,Optional=Optional,ListNode=ListNode,TreeNode=TreeNode,Node=Node)
                exec(compile(tree,str(path),'exec'),namespace)
                if solution is not None: namespace['s'] = namespace['Solution']()
                exec(test,namespace)
                behavior += 1
            except Exception as error:
                errors.append(f'{path.relative_to(ROOT)} block {index}: {type(error).__name__}: {error}')
    for error in errors: print(error)
    print(f'{syntax} Python examples parse; {behavior} documented variants pass behavior checks; {reference_examples} reference examples match their printed output; {contextual_examples} multi-file snippets checked for syntax only; {len(errors)} errors.')
    raise SystemExit(bool(errors))

if __name__ == '__main__': main()
