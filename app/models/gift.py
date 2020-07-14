from flask import current_app

from app.spider.yushu_book import YuShuBook
from sqlalchemy import Column, String, Integer, ForeignKey, Boolean, desc, func
from sqlalchemy.orm import relationship
from app.models.base import Base, db
from collections import namedtuple

EachGiftWishCount = namedtuple('EachGiftWishCount', ['count', 'isbn'])


class Gift(Base):
    __tablename__ = 'gift'

    id = Column(Integer, primary_key=True)
    uid = Column(Integer, ForeignKey('user.id'), nullable=False)  # 用户的唯一Id
    user = relationship('User')  # 表明礼物的从属关系
    isbn = Column(String(13), nullable=False)
    launched = Column(Boolean, default=False)  # 表明礼物是否赠送出去

    def is_yourself_gift(self, uid):
        return True if self.uid == uid else False

    @classmethod
    def get_user_gifts(cls, uid):
        gifts = Gift.query.filter_by(uid=uid, launched=False).order_by(
            desc(Gift.create_time)).all()
        return gifts

    @classmethod
    def get_wish_counts(cls, isbn_list):
        from app.models.wish import Wish

        # 根据传入的一组isbn，到Wish表中计算出某个礼物的Wish心愿数量
        # 条件表达式
        # mysql in 查询 isbn wish 的数量
        count_list = db.session.query(func.count(Wish.id), Wish.isbn).filter(
            Wish.launched == False,
            Wish.isbn.in_(isbn_list),
            Wish.status == 1).group_by(
            Wish.isbn).all()
        count_list = [{'count': w[0], 'isbn': w[1]} for w in count_list]
        # count_list = [EachGiftWishCount(w[0], w[1]) for w in count_list]
        return count_list

    @property
    def book(self):
        yushu_book = YuShuBook()
        yushu_book.search_by_isbn(self.isbn)
        return yushu_book.first

    # 对象代表一个礼物，具体
    # 类代表礼物这个事物，它是抽象，不是具体的“一个”
    @classmethod
    def recent(cls):
        # 链式调用
        # 主体query
        # 子函数
        # first() all()
        recent_gift = Gift.query.filter_by(
            launched=False).group_by(
            Gift.isbn).order_by(
            desc(Gift.create_time)).limit(
            current_app.config['RECENT_BOOK_COUNT']).distinct().all()
        return recent_gift

        # @classmethod
        # @cache.memoize(timeout=600)
        #  def recent(cls):
        #     gift_list = cls.query.filter_by(launched=False).order_by(
        #         desc(Gift.create_time)).group_by(Gift.book_id).limit(
        #         current_app.config['RECENT_BOOK_PER_PAGE']).all()
        #     view_model = GiftsViewModel.recent(gift_list)
        #     return view_model
