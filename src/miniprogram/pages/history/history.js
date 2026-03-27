// pages/history/history.js
var STORAGE_KEY = 'ai_detect_history';

Page({
  data: {
    sessions: [],
    expandedIds: {}   // { [id]: true } 控制折叠展开
  },

  onShow: function () {
    this.loadHistory();
  },

  // 加载历史记录
  loadHistory: function () {
    try {
      var raw = wx.getStorageSync(STORAGE_KEY);
      var sessions = raw ? JSON.parse(raw) : [];
      this.setData({ sessions: sessions });
    } catch (e) {
      this.setData({ sessions: [] });
    }
  },

  // 展开 / 折叠某条记录
  toggleSession: function (e) {
    var id = e.currentTarget.dataset.id;
    var expanded = Object.assign({}, this.data.expandedIds);
    expanded[id] = !expanded[id];
    this.setData({ expandedIds: expanded });
  },

  // 删除单条记录
  deleteSession: function (e) {
    var id = e.currentTarget.dataset.id;
    var that = this;
    wx.showModal({
      title: '确认删除',
      content: '删除这条历史记录？',
      success: function (res) {
        if (!res.confirm) return;
        try {
          var raw = wx.getStorageSync(STORAGE_KEY);
          var sessions = raw ? JSON.parse(raw) : [];
          sessions = sessions.filter(function (s) { return s.id !== id; });
          wx.setStorageSync(STORAGE_KEY, JSON.stringify(sessions));
          that.setData({ sessions: sessions });
        } catch (e) {}
      }
    });
  },

  // 清空全部历史
  clearAll: function () {
    var that = this;
    wx.showModal({
      title: '确认清空',
      content: '清空全部历史记录？此操作不可撤销。',
      success: function (res) {
        if (!res.confirm) return;
        wx.removeStorageSync(STORAGE_KEY);
        that.setData({ sessions: [], expandedIds: {} });
        wx.showToast({ title: '已清空', icon: 'success' });
      }
    });
  }
});
